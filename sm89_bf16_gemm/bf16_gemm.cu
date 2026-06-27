#include <cstdint>
#include <assert.h>
#include <cuda_bf16.h>

namespace gemm {

constexpr int WARP_SIZE = 32;

// SMEM_STRIDE in bytes, col in the units of 16-byte
template <int SMEM_STRIDE>
__device__ __forceinline__
int swizzle(int row, int col) {
    static_assert(SMEM_STRIDE == 128);
    // smem shape is [8r, 128B] => 8r rows and one row contains 128B
    // one row has 128B / 16B = 8 swizzle chunks => col in 0..8
    // (row % 8, col) in (0..8, 0..8)
    col ^= (row % 8);
    return row * SMEM_STRIDE + col * 16;
}

__device__ __forceinline__
void cp_async_commit_group() {
    asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ __forceinline__
void cp_async_wait_group() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

template <int HEIGHT, int WIDTH, int BLOCK_DIM>
__device__ __forceinline__
void cp_async_gmem_to_smem(int smem_dst, const nv_bfloat16* gmem_src, int stride) {
    // 一条 cp.async 指令，每个线程一次性搬运 16-byte，即 8 个 bf16
    constexpr int num_elements = 16 / sizeof(nv_bfloat16);

    // 一个 thread block 需要多少轮才能把 gmem_src[HEIGHT, WIDTH] 搬完
    constexpr int num_iters = (HEIGHT * WIDTH) / (BLOCK_DIM * num_elements);

    for (int iter = 0; iter < num_iters; ++iter) {
        // 在 iter 轮次，当前 thread 搬运的 16-byte 的一维偏移量，单位是元素个数
        const int offset = (iter * BLOCK_DIM + threadIdx.x) * num_elements;

        // 二维索引到 gmem_src[row, col]，其中 col 的单位是元素个数
        const int row = offset / WIDTH;
        const int col = offset % WIDTH;

        // swizzle 中 col 的单位必须 16-byte
        int smem_dst_addr = smem_dst + swizzle<WIDTH * sizeof(nv_bfloat16)>(row, col / num_elements);
        const nv_bfloat16* gmem_src_addr = gmem_src + row * stride + col;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_dst_addr), "l"(gmem_src_addr));
    }
}

__device__ __forceinline__
void ldmatrix_m8n8x4(int reg[4], int smem_addr) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "r"(smem_addr)
    );
}

__device__ __forceinline__
void mma_m16n8k16(const int A[4], const int B[2], float C[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "  // D: fp32x4
        "{%4, %5, %6, %7}, "  // A: bf16x8
        "{%8, %9}, "          // B: bf16x4
        "{%0, %1, %2, %3};"   // C: fp32x4
        : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
        "r"(B[0]), "r"(B[1])
    );
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int NUM_WARPS_M, int NUM_WARPS_N, int NUM_STAGES, int GROUP_M>
__global__ __launch_bounds__(NUM_WARPS_M * NUM_WARPS_N * WARP_SIZE)
void bf16_gemm_kernel(
    const nv_bfloat16* A, 
    const nv_bfloat16* B, 
    nv_bfloat16* C, 
    int M, int N, int K
) {
    constexpr int BLOCK_DIM = NUM_WARPS_M * NUM_WARPS_N * WARP_SIZE;
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    
    constexpr int WARP_M = BLOCK_M / NUM_WARPS_M;
    constexpr int WARP_N = BLOCK_N / NUM_WARPS_N;
    
    constexpr int NUM_MMA_M = WARP_M / MMA_M;
    constexpr int NUM_MMA_N = WARP_N / MMA_N;
    constexpr int NUM_MMA_K = BLOCK_K / MMA_K;

    const int bid = blockIdx.x;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    const int num_blocks_m = M / BLOCK_M;
    const int num_blocks_n = N / BLOCK_N;

    // TODO: threadblock swizzling to improve L2 cache hit rate
    int bid_m, bid_n;
    if constexpr (GROUP_M == 0) {
        // no threadblock swizzling
        bid_m = bid / num_blocks_n;
        bid_n = bid % num_blocks_n;
    } else {
        // threadblock swizzling to improve L2 cache hit rate
        // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
        // each group is [GROUP_M, num_blocks_n], tile from top (small M) to bottom (large M).
        // the last group might be shorter than GROUP_M if num_blocks_m % GROUP_M != 0.
        const int group_size = GROUP_M * num_blocks_n;
        const int group_id = bid / group_size;
        const int group_offset_m = group_id * GROUP_M;
        const int group_m = min(num_blocks_m - group_offset_m, GROUP_M);

        bid_m = group_offset_m + (bid % group_size) % group_m;
        bid_n = (bid % group_size) / group_m;
    }

    const int block_offset_m = bid_m * BLOCK_M;
    const int block_offset_n = bid_n * BLOCK_N;
    
    const int warp_id_m = warp_id / NUM_WARPS_N;
    const int warp_id_n = warp_id % NUM_WARPS_N;
    const int warp_offset_m = warp_id_m * WARP_M;
    const int warp_offset_n = warp_id_n * WARP_N;

    // A is row-major, B is column-major, C is row-major
    A += block_offset_m * K;
    B += block_offset_n * K;
    C += (block_offset_m + warp_offset_m) * N + (block_offset_n + warp_offset_n);

    // convert shared memory address to 32-bit from the start
    extern __shared__ nv_bfloat16 smem_ptr[];
    const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
    const int A_smem = smem;
    const int B_smem = A_smem + BLOCK_M * BLOCK_K * sizeof(nv_bfloat16);
    constexpr int AB_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16);

    int A_reg[NUM_MMA_K][NUM_MMA_M][4];
    int B_reg[NUM_MMA_K][NUM_MMA_N][2];
    float acc[NUM_MMA_M][NUM_MMA_N][4] = {};

    // pre-compute address used for ldmatrix, also pre-compute swizzling
    // lane in 0...8  => (0..8, 0)
    // lane in 8...16 => (8..16, 0)
    // lane in 16..24 => (0..8, 1)
    // lane in 24..32 => (8..16, 1)
    const int A_smem_ptr = A_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>(
        warp_offset_m + (lane_id % 16),
        (lane_id / 16)
    );

    // lane in 0...8  => (0..8, 0)
    // lane in 8...16 => (0..8, 1)
    // lane in 16..24 => (8..16, 0)
    // lane in 24..32 => (8..16, 1)
    const int B_smem_ptr = B_smem + swizzle<BLOCK_K * sizeof(nv_bfloat16)>(
        warp_offset_n + (lane_id % 8) + (lane_id / 16) * 8,
        (lane_id % 16) / 8
    );

    auto load_AB_gmem_to_smem = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;

        // A[BLOCK_M, BLOCK_K] -> A_smem[stage_id][BLOCK_M, BLOCK_K]
        // B[BLOCK_N, BLOCK_K] -> B_smem[stage_id][BLOCK_N, BLOCK_K]
        cp_async_gmem_to_smem<BLOCK_M, BLOCK_K, BLOCK_DIM>(A_smem + (stage_id * AB_size), A, K);
        cp_async_gmem_to_smem<BLOCK_N, BLOCK_K, BLOCK_DIM>(B_smem + (stage_id * AB_size), B, K);
        cp_async_commit_group();
        
        A += BLOCK_K;
        B += BLOCK_K;
    };

    /*
    MMA_K x bf16 => (16 * 2B) / 16B = 2 swizzle chunks => col in (0, 1)
    k -> k + 1 等价于在 MMA_K 方向 col -> col + 2k，即 swizzle(row, col) -> swizzle(row, col + 2k)
    要证明 swizzle(row, col + 2k) = swizzle(row, col) ^ 32k
    令 r = row % 8
    注意到 k in (0, 1, 2, 3) => 2k in (0, 2, 4, 6)
    那么 (col + 2k) in 0..8，且 (col + 2k) = (col ^ 2k)
    则 swizzle(row, col + 2k)
        = row * 128 + ((col + 2k) ^ r) * 16
        = row * 128 + ((col ^ 2k) ^ r) * 16
        = row * 128 + ((col ^ r) ^ 2k) * 16
    
    因为 ((col ^ r) * 16) ^ 32k
        = ((col ^ r) * 16) ^ ((2k) * 16)
        = ((col ^ r) ^ 2k) * 16
    
    所以 swizzle(row, col + 2k)
        = row * 128 + ((col ^ r) * 16) ^ 32k
        = swizzle(row, col) ^ 32k
    */
    auto compute = [&](int k_iter) {
        const int stage_id = k_iter % NUM_STAGES;

        // A_smem[stage_id][BLOCK_M, BLOCK_K] -> A_reg[NUM_MMA_K][NUM_MMA_M][4]
        for (int k = 0; k < NUM_MMA_K; ++k) {
            for (int m = 0; m < NUM_MMA_M; ++m) {
                int A_smem_addr = A_smem_ptr + (stage_id * AB_size) + m * MMA_M * BLOCK_K * sizeof(nv_bfloat16);
                ldmatrix_m8n8x4(A_reg[k][m], A_smem_addr ^ (k * 32));
            }
        }

        // B_smem[stage_id][BLOCK_N, BLOCK_K] -> B_reg[NUM_MMA_K][NUM_MMA_N][2]
        for (int k = 0; k < NUM_MMA_K; ++k) {
            for (int n = 0; n < NUM_MMA_N; n += 2) {
                int B_smem_addr = B_smem_ptr + (stage_id * AB_size) + n * MMA_N * BLOCK_K * sizeof(nv_bfloat16);
                ldmatrix_m8n8x4(B_reg[k][n], B_smem_addr ^ (k * 32));
            }
        }

        for (int k = 0; k < NUM_MMA_K; ++k) {
            for (int m = 0; m < NUM_MMA_M; ++m) {
                for (int n = 0; n < NUM_MMA_N; ++n) {
                    mma_m16n8k16(A_reg[k][m], B_reg[k][n], acc[m][n]);
                }
            }
        }
    };

    // initiate NUM_STAGES - 1 stages
    for (int stage_id = 0; stage_id < NUM_STAGES - 1; ++stage_id) {
        load_AB_gmem_to_smem(stage_id);
    }

    // loop invariance: there is always NUM_STAGES - 1 prefetch stages in-flight
    const int num_k_iters = K / BLOCK_K;
    for (int k_iter = 0; k_iter < num_k_iters - (NUM_STAGES - 1); ++k_iter) {
        // wait last MMA empty
        __syncthreads();

        // cp.async prefetch
        load_AB_gmem_to_smem(k_iter + NUM_STAGES - 1);

        // wait cp.async
        cp_async_wait_group<NUM_STAGES - 1>();

        // cp.async complete for all threads and do MMA
        __syncthreads();
        compute(k_iter);
    }

    for (int k_iter = num_k_iters - (NUM_STAGES - 1); k_iter < num_k_iters; ++k_iter) {
        // 冗余 commit_group
        cp_async_commit_group();

        // 原本是 num_k_iters - k_iter，现在是编译期常量 NUM_STAGES - 1
        cp_async_wait_group<NUM_STAGES - 1>();

        // cp.async complete for all threads and do MMA
        __syncthreads();
        compute(k_iter);
    }

    /*
    C[0] -> row = lane_id / 4,      col = (lane_id % 4) * 2 + 0
    C[1] -> row = lane_id / 4,      col = (lane_id % 4) * 2 + 1
    C[2] -> row = lane_id / 4 + 8,  col = (lane_id % 4) * 2 + 0
    C[3] -> row = lane_id / 4 + 8,  col = (lane_id % 4) * 2 + 1
    */
    for (int m = 0; m < NUM_MMA_M; ++m) {
        for (int n = 0; n < NUM_MMA_N; ++n) {
            const int row = m * MMA_M + (lane_id / 4);
            const int col = n * MMA_N + (lane_id % 4) * 2;
            float *c_reg = acc[m][n];
            reinterpret_cast<nv_bfloat162*>(C + ((row + 0) * N + col))[0] = __float22bfloat162_rn({c_reg[0], c_reg[1]});
            reinterpret_cast<nv_bfloat162*>(C + ((row + 8) * N + col))[0] = __float22bfloat162_rn({c_reg[2], c_reg[3]});
        }
    }
}

void launch_bf16_gemm_kernel(
    const nv_bfloat16* A, 
    const nv_bfloat16* B, 
    nv_bfloat16* C, 
    int M, int N, int K, 
    cudaStream_t stream
) {
    constexpr int BLOCK_M = 128;
    constexpr int BLOCK_N = 128;
    constexpr int BLOCK_K = 64;
    constexpr int NUM_WARPS_M = 2;
    constexpr int NUM_WARPS_N = 2;
    constexpr int NUM_STAGES = 1;
    constexpr int GROUP_M = 0;

    const int block_num = (M / BLOCK_M) * (N / BLOCK_N);
    const int thread_num = NUM_WARPS_M * NUM_WARPS_N * WARP_SIZE;
    const int smem_size = (BLOCK_M + BLOCK_N) * BLOCK_K * sizeof(nv_bfloat16) * NUM_STAGES;

    auto kernel = bf16_gemm_kernel<BLOCK_M, BLOCK_N, BLOCK_K, NUM_WARPS_M, NUM_WARPS_N, NUM_STAGES, GROUP_M>;
    if (smem_size > 48'000) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    }
    kernel<<<block_num, thread_num, smem_size, stream>>>(A, B, C, M, N, K);
}

}  // namespace gemm