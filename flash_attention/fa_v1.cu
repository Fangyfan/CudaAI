#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cub/block/block_reduce.cuh>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS) {                                 \
            std::cerr << "cuBLAS Error: " << static_cast<int>(status)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

namespace bench_utils {

__global__ void init_lm_kernel(float* l, float* m, int total_rows) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_rows) {
        l[idx] = 0.0f;
        m[idx] = -INFINITY;
    }
}

void reset_online_softmax_state(float* O, float* l, float* m,
                                int total_od, int total_rows,
                                cudaStream_t stream) {
    CHECK_CUDA(cudaMemsetAsync(O, 0, static_cast<size_t>(total_od) * sizeof(float), stream));
    constexpr int kBlock = 256;
    int grid = (total_rows + kBlock - 1) / kBlock;
    init_lm_kernel<<<grid, kBlock, 0, stream>>>(l, m, total_rows);
    CHECK_CUDA(cudaGetLastError());
}

struct CompareResult {
    bool pass;
    float max_abs_err;
    float max_rel_err;
    int idx;
};

CompareResult compare_tensors(const float* ref, const float* test, size_t n,
                              float abs_tol, float rel_tol) {
    CompareResult r{true, 0.0f, 0.0f, -1};
    for (size_t i = 0; i < n; ++i) {
        float abs_err = std::fabs(ref[i] - test[i]);
        float denom = std::max(std::fabs(ref[i]), 1e-6f);
        float rel_err = abs_err / denom;
        if (abs_err > r.max_abs_err || rel_err > r.max_rel_err) {
            r.max_abs_err = std::max(r.max_abs_err, abs_err);
            r.max_rel_err = std::max(r.max_rel_err, rel_err);
            r.idx = static_cast<int>(i);
        }
        if (abs_err > abs_tol && rel_err > rel_tol) {
            r.pass = false;
        }
    }
    return r;
}

template <typename PrepareFn, typename LaunchFn>
float benchmark_latency_ms(PrepareFn prepare,
                           LaunchFn launch,
                           int warmup_iters,
                           int bench_iters,
                           cudaStream_t stream) {
    for (int i = 0; i < warmup_iters; ++i) {
        prepare();
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < bench_iters; ++i) {
        prepare();
        CHECK_CUDA(cudaEventRecord(start, stream));
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / bench_iters;
}

}  // namespace bench_utils

namespace baseline {

template <int BLOCK_DIM>
__global__ void softmax_kernel(const float* input, float* output, int size, float scale) {
    input += blockIdx.x * size;
    output += blockIdx.x * size;

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_val;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        max_val = fmaxf(max_val, scale * input[i]);
    }
    max_val = BlockReduce(temp_storage).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum_val = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        sum_val += __expf(scale * input[i] - max_val);
    }
    sum_val = BlockReduce(temp_storage).Reduce(sum_val, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum_val;
    }
    __syncthreads();
    sum_val = shared_val;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = __expf(scale * input[i] - max_val) / sum_val;
    }
}

void launch_softmax(const float* input, float* output, int N, int C, float scale, cudaStream_t stream) {
    dim3 block_dim(512);
    dim3 grid_dim(N);
    softmax_kernel<512><<<grid_dim, block_dim, 0, stream>>>(input, output, C, scale);
}

void launch_attention_baseline(const float* Q, const float* K, const float* V, float* O, float* S, float* P,
                               int batch, int heads, int N, int d, cublasHandle_t handle, cudaStream_t stream) {
    const float scale = rsqrtf(static_cast<float>(d));
    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSetStream(handle, stream));

    // S = Q * K^T, shape: [batch * heads, N, N]
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_T,
                                           CUBLAS_OP_N,
                                           N, N, d,
                                           &alpha,
                                           K, d, N * d,
                                           Q, d, N * d,
                                           &beta,
                                           S, N, N * N,
                                           batch * heads));

    // P = softmax(S / sqrt(d))
    launch_softmax(S, P, batch * heads * N, N, scale, stream);

    // O = P * V, shape: [batch * heads, N, d]
    CHECK_CUBLAS(cublasSgemmStridedBatched(handle,
                                           CUBLAS_OP_N,
                                           CUBLAS_OP_N,
                                           d, N, N,
                                           &alpha,
                                           V, d, N * d,
                                           P, N, N * N,
                                           &beta,
                                           O, d, N * d,
                                           batch * heads));
}

}  // namespace baseline

namespace fa1_minimal {
/*
Flash Attention Minimal: FlashAttention-V1 的 CUDA 实现
https://github.com/tspeterkim/flash-attention-minimal

gridDim(batch, heads)
blockDim(Bc)
Q \ K \ V \ O: [batch, heads, N, d]
l \ m: [batch, heads, N, 1]
*/
__global__ void flash_attention_minimal_kernel(const float *Q, const float *K, const float *V,
                                        float *O, float* l, float* m,
                                        const int N, const int d,
                                        const int Tc, const int Tr,
                                        const int Bc, const int Br,
                                        const float scale) {
    int tx = threadIdx.x;
    int bx = blockIdx.x; // batch_id
    int by = blockIdx.y; // head_id

    int qkv_offset = (bx * gridDim.y * N * d) + (by * N * d);
    int lm_offset  = (bx * gridDim.y * N) + (by * N);

    extern __shared__ float sram[];
    int tile_size = Bc * d; // size of Qi, Kj, Vj 必须确保 Br = Bc
    float *Qi = sram;
    float *Kj = &sram[tile_size];
    float *Vj = &sram[tile_size * 2];
    float *S  = &sram[tile_size * 3]; // Bc * Br

    // 外层循环 j：遍历 K 和 V 块
    for (int j = 0; j < Tc; j++) {
        // Load Kj, Vj to SRAM
        for (int x = 0; x < d; x++) {
            Kj[(tx * d) + x] = K[qkv_offset + (tile_size * j) + (tx * d) + x]; // Kj[tx][...]
            Vj[(tx * d) + x] = V[qkv_offset + (tile_size * j) + (tx * d) + x]; // Vj[tx][...]
        }
        __syncthreads();

        // 内层循环 i：遍历 Q 块与 Online Softmax 计算
        for (int i = 0; i < Tr; i++) {
            // Load Qi to SRAM
            for (int x = 0; x < d; x++) {
                Qi[(tx * d) + x] = Q[qkv_offset + (tile_size * i) + (tx * d) + x]; // Qi[tx][...]
            }

            // j == 0 时直接使用在线 softmax 的初始值，避免外部额外初始化
            float row_m_prev = (j == 0) ? -INFINITY : m[lm_offset + (Br * i) + tx];
            float row_l_prev = (j == 0) ? 0.0f      : l[lm_offset + (Br * i) + tx];

            // S = QK^T, row_m = rowmax(S)
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++) {
                float sum = 0.0f;
                for (int x = 0; x < d; x++) {
                    sum += Qi[(tx * d) + x] * Kj[(y * d) + x]; // S[tx][y] = Qi[tx][...] * Kj[y][...]
                }
                sum *= scale;
                S[(Bc * tx) + y] = sum; // S[tx][y]
                if (sum > row_m) row_m = sum;
            }

            // P = exp(S - row_m), row_l = rowsum(P)
            float row_l = 0.0f;
            for (int y = 0; y < Bc; y++) {
                S[(Bc * tx) + y] = __expf(S[(Bc * tx) + y] - row_m); // S[tx][y] = e^{S[tx][y] - row_m}
                row_l += S[(Bc * tx) + y]; // row_l = \sum S[tx][...]
            }

            // Compute new m and l
            // new_m = max(old_m, row_m)
            // new_l = e^{old_m - new_m} * old_l + e^{row_m - new_m} * row_l
            float row_m_new = fmaxf(row_m_prev, row_m);
            float row_l_new = (__expf(row_m_prev - row_m_new) * row_l_prev) + (__expf(row_m - row_m_new) * row_l);

            // Write O, l, m to HBM
            for (int x = 0; x < d; x++) {
                float pv = 0.0f;
                for (int y = 0; y < Bc; y++) {
                    pv += S[(Bc * tx) + y] * Vj[(y * d) + x]; // pv[tx][x] = S[tx][y] * Vj[y][x]
                }

                // O[tx][x] = (1 / new_l) * (e^{old_m - new_m} * old_l * O[tx][x] + e^{row_m - new_m} * pv[tx][x])
                float old_o = (j == 0) ? 0.0f : O[qkv_offset + (tile_size * i) + (tx * d) + x];
                O[qkv_offset + (tile_size * i) + (tx * d) + x] = (1.0f / row_l_new) * (
                    (row_l_prev * __expf(row_m_prev - row_m_new) * old_o) + 
                    (__expf(row_m - row_m_new) * pv)
                );
            }

            m[lm_offset + (Br * i) + tx] = row_m_new;
            l[lm_offset + (Br * i) + tx] = row_l_new;
        }
        __syncthreads();
    }
}

void launch_flash_attention_minimal(const float* Q, const float* K, const float* V, float* O, float* l, float* m,
                                    const int batch, const int heads, const int N, const int d, cudaStream_t stream) {
    constexpr int Bc = 16;
    constexpr int Br = 16;
    const int Tr = N / Br;
    const int Tc = N / Bc;
    const float scale = rsqrtf(static_cast<float>(d));

    if (N % Bc != 0 || N % Br != 0) {
        std::cerr << "Error: N must be divisible by Bc and Br." << std::endl;
        std::exit(EXIT_FAILURE);
    }

    const int sram_size = (3 * Bc * d + Bc * Br) * sizeof(float);

    dim3 grid_dim(batch, heads);
    dim3 block_dim(Bc);
    flash_attention_minimal_kernel<<<grid_dim, block_dim, sram_size, stream>>>(Q, K, V, O, l, m, N, d, Tc, Tr, Bc, Br, scale);
}

}  // namespace minimal

namespace fa1_version1 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

/*
每次循环处理一行 Q[1, d] 的计算任务
grid(heads, batch)
block(BLOCK_DIM = 128)
Q / O: [batch, heads, N, d]
K / V: [batch, heads, N, d]
l / m: [batch, heads, N, 1]
*/
template <int BLOCK_DIM, int Bc>
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, float* l, float* m, int N, int d, float scale) {
    int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

    extern __shared__ float sram[];
    float* s_Q = sram;          // [1, d]
    float* s_K = s_Q + d;       // [Bc, d]
    float* s_V = s_K + Bc * d;  // [Bc, d]
    float* S = s_V + Bc * d;    // [1, Bc]

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ MD row_ml_old, shared_temp_ml;

    // 对 KV 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int j = 0; j < N; j += Bc) {
        // 加载 [Bc, d] 数据到 s_K 和 s_V
        for (int i = threadIdx.x; i < Bc * d; i += blockDim.x) {
            s_K[i] = K[kv_offset + j * d + i];
            s_V[i] = V[kv_offset + j * d + i];
        }
        __syncthreads();

        // 遍历 Q 的 N 行，每次处理一行 [1, d]
        for (int i = 0; i < N; ++i) {
            // 加载 1 行数据 [1, d] 到 s_Q
            for (int k = threadIdx.x; k < d; k += blockDim.x) {
                s_Q[k] = Q[qo_offset + i * d + k];
            }
            // 上一组 KV[Bc, d] 结束时每行的 m 和 l
            if (threadIdx.x == 0) {
                row_ml_old = { m[lm_offset + i], l[lm_offset + i] };
            }
            __syncthreads();

            // 存储当前第 i 行的 m 和 l
            MD row_ml = { -INFINITY, 0.0f };
            // 遍历 K^T 的 Bc 行
            for (int k = 0; k < Bc; ++k) {
                MD temp_ml = { 0.0f, 1.0f };
                // 计算 QK^T
                for (int x = threadIdx.x; x < d; x += blockDim.x) {
                    temp_ml.m += s_Q[x] * s_K[k * d + x];
                }
                temp_ml.m *= scale;
                temp_ml.m = BlockReduce(tempStorage).Reduce(temp_ml.m, cub::Sum());
                if (threadIdx.x == 0) {
                    S[k] = temp_ml.m;
                    shared_temp_ml = temp_ml;
                }
                __syncthreads();
                temp_ml = shared_temp_ml;
                row_ml = MD_OP()(row_ml, temp_ml);
            }
            MD row_ml_new = MD_OP()(row_ml_old, row_ml);

            // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
            for (int k = threadIdx.x; k < d; k += blockDim.x) {
                float pv = 0.0f;
                for (int x = 0; x < Bc; ++x) {
                    pv += __expf(S[x] - row_ml.m) * s_V[x * d + k];
                }
                // 更新 O 矩阵
                O[qo_offset + i * d + k] = (1.0f / row_ml_new.d) * (
                    row_ml_old.d * __expf(row_ml_old.m - row_ml_new.m) * O[qo_offset + i * d + k] + 
                    __expf(row_ml.m - row_ml_new.m) * pv
                );
            }

            // 写入当前组 KV[Bc, d] 的 l 和 m
            if (threadIdx.x == 0) {
                l[lm_offset + i] = row_ml_new.d;
                m[lm_offset + i] = row_ml_new.m;
            }
            __syncthreads();
        }
    }
}

void launch_flash_attention(float* Q, float* K, float* V, float* O, float* l, float* m, int batch, int heads, int N, int d, cudaStream_t stream) {
    constexpr int Bc = 16;
    constexpr int BLOCK_DIM = 128;
    float scale = rsqrtf(d);
    int sram_size = (d + 2 * Bc * d + Bc) * sizeof(float);
    dim3 gridDim(heads, batch);
    dim3 blockDim(BLOCK_DIM);
    flash_attention_kernel<BLOCK_DIM, Bc><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, l, m, N, d, scale);
}

}  // namespace fa1_version1

namespace fa1_version2 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, delta);
    }
    return val;
}

/*
每次循环处理 Br 行 Q[Br, d] 的计算任务
grid(heads, batch)
block(BLOCK_DIM = 32 * Br)
Q / O: [batch, heads, N, d]
K / V: [batch, heads, N, d]
l / m: [batch, heads, N, 1]
*/
template <int Br, int Bc>
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, float* l, float* m, int N, int d, float scale) {
    int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    extern __shared__ float sram[];
    float* s_Q = sram;          // [Br, d]
    float* s_K = s_Q + Br * d;  // [Bc, d]
    float* s_V = s_K + Bc * d;  // [Bc, d]
    float* S = s_V + Bc * d;    // [Br, Bc]

    __shared__ MD row_ml_old[Br];

    // 对 KV 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int j = 0; j < N; j += Bc) {
        // 加载 [Bc, d] 数据到 s_K 和 s_V
        for (int i = threadIdx.x; i < Bc * d; i += blockDim.x) {
            s_K[i] = K[kv_offset + j * d + i];
            s_V[i] = V[kv_offset + j * d + i];
        }
        __syncthreads();

        // 遍历 Q 的 N 行，每次处理 Br 行 [Br, d]
        for (int i = 0; i < N; i += Br) {
            // 加载 Br 行数据 [Br, d] 到 s_Q
            for (int k = threadIdx.x; k < Br * d; k += blockDim.x) {
                s_Q[k] = Q[qo_offset + i * d + k];
            }
            // 上一组 KV[Bc, d] 结束时每行的 m 和 l
            if (threadIdx.x < Br) {
                row_ml_old[threadIdx.x] = { m[lm_offset + i + threadIdx.x], l[lm_offset + i + threadIdx.x] };
            }
            __syncthreads();

            // 存储当前 warp 对应的第 i + warp_id 行的 m 和 l
            MD row_ml = { -INFINITY, 0.0f };
            // 遍历 K^T 的 Bc 行
            for (int k = 0; k < Bc; ++k) {
                MD temp_ml = { 0.0f, 1.0f };
                // 当前 warp 内每个线程计算部分点积 QK^T
                for (int x = lane; x < d; x += 32) {
                    temp_ml.m += s_Q[warp * d + x] * s_K[k * d + x];
                }
                temp_ml.m *= scale;
                __syncwarp();

                // 存储第 i + warp_id 行的 Q 向量与第 k 列的 s_K 向量的内积, QK^T 矩阵当前第 warp_id 行的值
                temp_ml.m = warp_reduce_sum(temp_ml.m);
                row_ml = MD_OP()(row_ml, temp_ml);
                if (lane == 0) {
                    S[warp * Bc + k] = temp_ml.m;
                }
            }
            __syncthreads();
            MD row_ml_new = MD_OP()(row_ml_old[warp], row_ml);

            // 遍历矩阵 O 的 d 维度，O = softmax(QK^T)V
            for (int k = lane; k < d; k += 32) {
                float pv = 0.0f;
                for (int x = 0; x < Bc; ++x) {
                    pv += __expf(S[warp * Bc + x] - row_ml.m) * s_V[x * d + k];
                }
                // 更新 O 矩阵
                O[qo_offset + (i + warp) * d + k] = (1.0f / row_ml_new.d) * (
                    row_ml_old[warp].d * __expf(row_ml_old[warp].m - row_ml_new.m) * O[qo_offset + (i + warp) * d + k] + 
                    __expf(row_ml.m - row_ml_new.m) * pv
                );
            }

            // 写入当前组 KV[Bc, d] 的 l 和 m
            if (lane == 0) {
                l[lm_offset + i + warp] = row_ml_new.d;
                m[lm_offset + i + warp] = row_ml_new.m;
            }
            __syncthreads();
        }
    }
}

void launch_flash_attention(float* Q, float* K, float* V, float* O, float* l, float* m, int batch, int heads, int N, int d, cudaStream_t stream) {
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int BLOCK_DIM = Br * 32;
    float scale = rsqrtf(d);
    int sram_size = (Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);
    dim3 gridDim(heads, batch);
    dim3 blockDim(BLOCK_DIM);
    flash_attention_kernel<Br, Bc><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, l, m, N, d, scale);
}

}  // namespace fa1_version2

namespace fa1_version3 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

__device__ __forceinline__ MD shfl_xor_val(MD val, int delta) {
    MD res;
    res.m = __shfl_xor_sync(0xffffffff, val.m, delta);
    res.d = __shfl_xor_sync(0xffffffff, val.d, delta);
    return res;
}

__device__ __forceinline__ MD warp_reduce_md(MD val, MD_OP op) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = op(val, shfl_xor_val(val, delta));
    }
    return val;
}

// load_QK_from_gmem_and_convert_to_half<Br, Bc, Bd>(Q, K, d, s_Q_half, s_K_half, qo_offset + i * d + k, kv_offset + j * d + k);
template <int Br, int Bc, int Bd>
__device__ void load_QK_from_gmem_and_convert_to_half(float* Q, float* K, int d, half* s_Q, half* s_K, int offset_q, int offset_k) {
    int row, col;
    float4 temp4;

    // 每个线程负责 4 个元素
    for (int i = (threadIdx.x << 2); i < Br * Bd; i += (blockDim.x << 2)) {
        row = i / Bd;
        col = i % Bd;
        temp4 = reinterpret_cast<float4*>(Q + offset_q + row * d + col)[0];
        s_Q[row * Bd + col] = __float2half(temp4.x);
        s_Q[row * Bd + col + 1] = __float2half(temp4.y);
        s_Q[row * Bd + col + 2] = __float2half(temp4.z);
        s_Q[row * Bd + col + 3] = __float2half(temp4.w);
    }
    for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2)) {
        row = i / Bd;
        col = i % Bd;
        temp4 = reinterpret_cast<float4*>(K + offset_k + row * d + col)[0];
        s_K[row * Bd + col] = __float2half(temp4.x);
        s_K[row * Bd + col + 1] = __float2half(temp4.y);
        s_K[row * Bd + col + 2] = __float2half(temp4.z);
        s_K[row * Bd + col + 3] = __float2half(temp4.w);
    }
}

// load_V_from_gmem_and_convert_to_half<Bc, Bd>(V, d, s_V_half, kv_offset + j * d + k);
template <int Bc, int Bd>
__device__ void load_V_from_gmem_and_convert_to_half(float* V, int d, half* s_V, int offset_v) {
    int row, col;
    float4 temp4;

    // 每个线程负责 4 个元素
    for (int i = (threadIdx.x << 2); i < Bc * Bd; i += (blockDim.x << 2)) {
        row = i / Bd;
        col = i % Bd;
        temp4 = reinterpret_cast<float4*>(V + offset_v + row * d + col)[0];
        s_V[row * Bd + col] = __float2half(temp4.x);
        s_V[row * Bd + col + 1] = __float2half(temp4.y);
        s_V[row * Bd + col + 2] = __float2half(temp4.z);
        s_V[row * Bd + col + 3] = __float2half(temp4.w);
    }
}

// gemm_QK_from_smem_by_wmma<Bd, Wr, Wc, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragBType, FragCFloatType>(
//      s_Q_half, s_K_half, a_frag, b_frag, acc_frag, warp_row, warp_col);
template <int Bd, int Wr, int Wc, int WM_ITERS, int WN_ITERS, int WK_ITERS, typename T1, typename T2, typename T3>
__device__ void gemm_QK_from_smem_by_wmma(half* s_Q, half* s_K, T1* q_frag, T2* k_frag, T3* acc_frag, int warp_row, int warp_col) {
    using namespace nvcuda; // wmma

#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
            int offset = (warp_row * Wr + wm_idx * 16) * Bd + (wk_idx * 16);
            wmma::load_matrix_sync(q_frag[wm_idx * WK_ITERS + wk_idx], s_Q + offset, Bd);
        }
    }

#pragma unroll
    for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
        for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
            int offset = (warp_col * Wc + wn_idx * 16) * Bd + (wk_idx * 16);
            wmma::load_matrix_sync(k_frag[wn_idx * WK_ITERS + wk_idx], s_K + offset, Bd);
        }
    }

#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
                wmma::mma_sync(
                    acc_frag[wm_idx * WN_ITERS + wn_idx], 
                    q_frag[wm_idx * WK_ITERS + wk_idx], 
                    k_frag[wn_idx * WK_ITERS + wk_idx], 
                    acc_frag[wm_idx * WN_ITERS + wn_idx]
                );
            }
        }
    }
}

// gemm_PV_from_smem_by_wmma<Bd, Wr, Wc, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragVType, FragCFloatType>(
//     s_V_half, a_frag, v_frag, acc_frag, warp_row, warp_col);
template <int Bd, int Wr, int Wc, int WM_ITERS, int WN_ITERS, int WK_ITERS, typename T1, typename T2, typename T3>
__device__ void gemm_PV_from_smem_by_wmma(half* s_V, T1* p_frag, T2* v_frag, T3* acc_frag, int warp_row, int warp_col) {
    using namespace nvcuda; // wmma

#pragma unroll
    for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
        for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
            int offset = (wk_idx * 16) * Bd + (warp_col * Wc + wn_idx * 16);
            wmma::load_matrix_sync(v_frag[wn_idx * WK_ITERS + wk_idx], s_V + offset, Bd);
        }
    }

#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
                wmma::mma_sync(
                    acc_frag[wm_idx * WN_ITERS + wn_idx], 
                    p_frag[wm_idx * WK_ITERS + wk_idx], 
                    v_frag[wn_idx * WK_ITERS + wk_idx], 
                    acc_frag[wm_idx * WN_ITERS + wn_idx]
                );
            }
        }
    }
}

// load_S_half_from_smem_to_frag<Bc, Wr, Wc, WM_ITERS, WK_ITERS, FragAType>(s_S_half, a_frag, warp_row, warp_col);
template <int Bc, int Wr, int Wc, int WM_ITERS, int WK_ITERS, typename T>
__device__ void load_S_half_from_smem_to_frag(half* s_S, T* p_frag, int warp_row, int warp_col) {
    using namespace nvcuda;
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
        for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
            int offset = (warp_row * Wr + wm_idx * 16) * Bc + (wk_idx * 16);
            wmma::load_matrix_sync(p_frag[wm_idx * WK_ITERS + wk_idx], s_S + offset, Bc);
        }
    }
}

// store_gemm_S_to_smem<Bc, Wr, Wc, WM_ITERS, WN_ITERS, FragCFloatType>(s_S, acc_frag, warp_row, warp_col);
template <int Bc, int Wr, int Wc, int WM_ITERS, int WN_ITERS, typename T>
__device__ void store_gemm_S_to_smem(float* s_S, T* acc_frag, int warp_row, int warp_col) {
    using namespace nvcuda;
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
            int offset = (warp_row * Wr + wm_idx * 16) * Bc + (warp_col * Wc + wn_idx * 16);
            wmma::store_matrix_sync(s_S + offset, acc_frag[wm_idx * WN_ITERS + wn_idx], Bc, wmma::mem_row_major);
        }
    }
}

// store_gemm_O_to_smem<Bd, Wr, Wc, WM_ITERS, WN_ITERS, FragCFloatType>(s_O, acc_frag, warp_row, warp_col);
template <int Bd, int Wr, int Wc, int WM_ITERS, int WN_ITERS, typename T>
__device__ void store_gemm_O_to_smem(float* s_O, T* acc_frag, int warp_row, int warp_col) {
    using namespace nvcuda;
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
            int offset = (warp_row * Wr + wm_idx * 16) * Bd + (warp_col * Wc + wn_idx * 16);
            wmma::store_matrix_sync(s_O + offset, acc_frag[wm_idx * WN_ITERS + wn_idx], Bd, wmma::mem_row_major);
        }
    }
}

/*
每次循环处理 Br 行 Q[Br, d] 的计算任务，在 d 维度上更细粒度的分块计算
grid(heads, batch)
block(BLOCK_DIM)
Q / O: [batch, heads, N, d]
K / V: [batch, heads, N, d]
l / m: [batch, heads, N, 1]
*/
template <int Br, int Bc, int Bd, int Wr, int Wc, int WM_ITERS, int WN_ITERS, int WK_ITERS>
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, float* l, float* m, int N, int d, float scale) {
    using namespace nvcuda; // wmma
    auto op = MD_OP();

    // 当前矩阵偏移量
    int qo_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    int kv_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N * d;
    int lm_offset = (blockIdx.y * gridDim.x + blockIdx.x) * N;

    __shared__ half s_Q_half[Br * Bd];
    __shared__ half s_K_half[Bc * Bd];
    __shared__ half s_V_half[Bc * Bd];
    __shared__ half s_S_half[Br * Bc];
    __shared__ float s_S[Br * Bc];
    __shared__ float s_O[Br * Bd];

    __shared__ MD row_ml[Br];
    __shared__ MD row_ml_old[Br];
    __shared__ MD row_ml_new[Br];

    // block 内 warp 的二维分布
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp_col = warp % (Bc / Wc);
    int warp_row = warp / (Bc / Wc);
    int warp_num = blockDim.x >> 5;

    using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major>;
    using FragCFloatType = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;
    using FragVType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;

    // 当前 warp 内的矩阵乘法片段
    FragAType a_frag[WM_ITERS * WK_ITERS];        // 用于存储左乘矩阵 Q[Br, Bd] 和 QK^T[Br, Bc] 的分片
    FragBType b_frag[WN_ITERS * WK_ITERS];        // 用于存储右乘矩阵 K[Bc, Bd] 的分片
    FragCFloatType acc_frag[WM_ITERS * WN_ITERS]; // 用于存储结果矩阵 QK^T[Br, Bc], PV[Br, Bd] 的分片
    FragVType v_frag[WN_ITERS * WK_ITERS];        // 用于存储右乘矩阵 V[Bc, Bd] 的分片

    // 对 KV 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int j = 0; j < N; j += Bc) {
        // 对 Q 在 N 维度分组，每组长度为 Br，共分为 Tr = N / Br 组
        for (int i = 0; i < N; i += Br) {
            // 上一组 KV[Bc, d] 结束时每行 (Q 的 i ... i + Br - 1 行) 对应的 m 和 l
            for (int r = threadIdx.x; r < Br; r += blockDim.x) {
                row_ml[r] = { -INFINITY, 0.0f };
                row_ml_old[r] = { m[lm_offset + (i + r)], l[lm_offset + (i + r)] };
            }
            __syncthreads();

            // 清空 acc_frag
            #pragma unroll
            for (int x = 0; x < WM_ITERS * WN_ITERS; ++x) {
                wmma::fill_fragment(acc_frag[x], 0.0f);
            }

            // 枚举 K 的第 k 组小块 s_K[Bc, Bd]，计算 QK^T[Br, Bc] 矩阵
            for (int k = 0; k < d; k += Bd) {
                load_QK_from_gmem_and_convert_to_half<Br, Bc, Bd>(
                    Q, K, d, s_Q_half, s_K_half, qo_offset + i * d + k, kv_offset + j * d + k
                );
                __syncthreads();

                gemm_QK_from_smem_by_wmma<Bd, Wr, Wc, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragBType, FragCFloatType>(
                    s_Q_half, s_K_half, a_frag, b_frag, acc_frag, warp_row, warp_col
                );
                __syncthreads();
            }
            store_gemm_S_to_smem<Bc, Wr, Wc, WM_ITERS, WN_ITERS, FragCFloatType>(s_S, acc_frag, warp_row, warp_col);
            __syncthreads();

            // 对 s_S[Br, Bc] 求 softmax，每个 warp 计算一行 (第 r 行)
            for (int r = warp; r < Br; r += warp_num) {
                MD row_ml_temp = { -INFINITY, 0.0f };
                // warp 内线程分工，对 s_S[Br, Bd] 进行 online softmax
                for (int c = lane; c < Bc; c += 32) {
                    MD ml_val = { scale * s_S[r * Bc + c], 1.0f };
                    row_ml_temp = op(row_ml_temp, ml_val);
                }
                __syncwarp();

                // 得到 s_S[Br, Bc] 每一行的 m 和 l
                row_ml_temp = warp_reduce_md(row_ml_temp, op);
                if (lane == 0) {
                    row_ml[r] = row_ml_temp;
                    row_ml_new[r] = op(row_ml_old[r], row_ml_temp);
                }

                // warp 内线程分工，更新 s_S_half[Br, Bc]
                for (int c = lane; c < Bc; c += 32) {
                    s_S_half[r * Bc + c] = __float2half(__expf(scale * s_S[r * Bc + c] - row_ml_temp.m));
                }
            }
            __syncthreads();

            load_S_half_from_smem_to_frag<Bc, Wr, Wc, WM_ITERS, WK_ITERS, FragAType>(s_S_half, a_frag, warp_row, warp_col);

            // 枚举 V 的第 k 组小块 s_V[Bc, Bd]，计算 s_S_half[Br, Bc] * s_V[Bc, Bd]
            for (int k = 0; k < d; k += Bd) {
                // 清空 acc_frag
                #pragma unroll
                for (int x = 0; x < WM_ITERS * WN_ITERS; ++x) {
                    wmma::fill_fragment(acc_frag[x], 0.0f);
                }

                load_V_from_gmem_and_convert_to_half<Bc, Bd>(V, d, s_V_half, kv_offset + j * d + k);
                __syncthreads();

                gemm_PV_from_smem_by_wmma<Bd, Wr, Wc, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragVType, FragCFloatType>(
                    s_V_half, a_frag, v_frag, acc_frag, warp_row, warp_col
                );

                store_gemm_O_to_smem<Bd, Wr, Wc, WM_ITERS, WN_ITERS, FragCFloatType>(s_O, acc_frag, warp_row, warp_col);
                __syncthreads();

                for (int r = warp; r < Br; r += warp_num) {
                    // warp 内线程分工，更新 O[Br, Bd]
                    for (int c = lane; c < Bd; c += 32) {
                        O[qo_offset + (i + r) * d + (k + c)] = (1.0f / row_ml_new[r].d) * (
                            row_ml_old[r].d * __expf(row_ml_old[r].m - row_ml_new[r].m) * O[qo_offset + (i + r) * d + (k + c)] + 
                            __expf(row_ml[r].m - row_ml_new[r].m) * s_O[r * Bd + c]
                        );
                    }
                }
            }

            // 写入当前组 KV[Bc, d] 的 l 和 m
            for (int r = threadIdx.x; r < Br; r += blockDim.x) {
                l[lm_offset + (i + r)] = row_ml_new[r].d;
                m[lm_offset + (i + r)] = row_ml_new[r].m;
            }
            __syncthreads();
        }
    }
}

void launch_flash_attention(float* Q, float* K, float* V, float* O, float* l, float* m, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    // 让 Bd = Bc 从而使得 S = QK^T 矩阵分片 [Br, Bc] 与 Q / O 矩阵分片 [Br, Bd] 形状相同，方便排布
    constexpr int Br = 64;
    constexpr int Bc = 32;
    constexpr int Bd = Bc;
    constexpr int Wr = 32;
    constexpr int Wc = 16;
    constexpr int BLOCK_DIM = (Br / Wr) * (Bc / Wc) * 32;
    // 单个 warp 处理矩阵乘法 [M,K] × [K,N] = [M,N] 层面 M、N、K 方向每个 warp 的迭代次数
    constexpr int WM_ITERS = Wr / 16;
    constexpr int WN_ITERS = Wc / 16;
    constexpr int WK_ITERS = Bd / 16;
    dim3 gridDim(heads, batch);
    dim3 blockDim(BLOCK_DIM);
    flash_attention_kernel<Br, Bc, Bd, Wr, Wc, WM_ITERS, WN_ITERS, WK_ITERS><<<gridDim, blockDim, 0, stream>>>(Q, K, V, O, l, m, N, d, scale);
}

}  // namespace fa1_version3
struct KernelReport {
    std::string name;
    float avg_ms;
    bool pass;
    float max_abs_err;
    float max_rel_err;
};

int main() {
    constexpr int batch = 4;
    constexpr int heads = 4;
    constexpr int N = 512;
    constexpr int d = 128;

    constexpr int warmup__iters = 10;
    constexpr int bench__iters  = 20;

    constexpr size_t qkv_elems = static_cast<size_t>(batch) * heads * N * d;
    constexpr size_t o_elems = qkv_elems;
    constexpr size_t rows = static_cast<size_t>(batch) * heads * N;
    constexpr size_t s_elems = static_cast<size_t>(batch) * heads * N * N;

    std::vector<float> h_Q(qkv_elems);
    std::vector<float> h_K(qkv_elems);
    std::vector<float> h_V(qkv_elems);
    std::vector<float> h_O_ref(o_elems);
    std::vector<float> h_O_test(o_elems);

    for (size_t i = 0; i < qkv_elems; ++i) {
        h_Q[i] = static_cast<float>(rand()) / RAND_MAX;
        h_K[i] = static_cast<float>(rand()) / RAND_MAX;
        h_V[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_Q = nullptr;
    float* d_K = nullptr;
    float* d_V = nullptr;
    float* d_O_ref = nullptr;
    float* d_O = nullptr;
    float* d_S = nullptr;
    float* d_P = nullptr;
    float* d_l = nullptr;
    float* d_m = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Q, qkv_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O_ref, o_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_O, o_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, s_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, s_elems * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_l, rows * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m, rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q.data(), qkv_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K.data(), qkv_elems * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V.data(), qkv_elems * sizeof(float), cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    auto baseline_prepare = [&]() {};
    auto baseline_launch = [&]() {
        baseline::launch_attention_baseline(d_Q, d_K, d_V, d_O_ref, d_S, d_P,
                                            batch, heads, N, d, handle, stream);
    };

    float baseline_ms = bench_utils::benchmark_latency_ms(
        baseline_prepare, baseline_launch, warmup__iters, bench__iters, stream);

    baseline_launch();
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaMemcpy(h_O_ref.data(), d_O_ref, o_elems * sizeof(float), cudaMemcpyDeviceToHost));

    std::vector<KernelReport> reports;
    reports.push_back({"baseline", baseline_ms, true, 0.0f, 0.0f});

    auto run_and_check = [&](const std::string& name,
                             auto&& prepare,
                             auto&& launch,
                             float abs_tol,
                             float rel_tol) {
        float avg_ms = bench_utils::benchmark_latency_ms(prepare, launch,
                                                         warmup__iters, bench__iters, stream);
        prepare();
        launch();
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUDA(cudaMemcpy(h_O_test.data(), d_O, o_elems * sizeof(float), cudaMemcpyDeviceToHost));
        auto cmp = bench_utils::compare_tensors(h_O_ref.data(), h_O_test.data(), o_elems,
                                                abs_tol, rel_tol);
        reports.push_back({name, avg_ms, cmp.pass, cmp.max_abs_err, cmp.max_rel_err});
    };

    auto prepare_online = [&]() {
        bench_utils::reset_online_softmax_state(d_O, d_l, d_m,
                                                static_cast<int>(o_elems),
                                                static_cast<int>(rows),
                                                stream);
    };

    run_and_check(
        "fa1_minimal",
        []() {},
        [&]() {
            fa1_minimal::launch_flash_attention_minimal(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                        batch, heads, N, d, stream);
        },
        3e-3f, 3e-3f);

    run_and_check(
        "fa1_version1",
        prepare_online,
        [&]() {
            fa1_version1::launch_flash_attention(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                 batch, heads, N, d, stream);
        },
        3e-3f, 3e-3f);

    run_and_check(
        "fa1_version2",
        prepare_online,
        [&]() {
            fa1_version2::launch_flash_attention(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                 batch, heads, N, d, stream);
        },
        3e-3f, 3e-3f);

    run_and_check(
        "fa1_version3",
        prepare_online,
        [&]() {
            fa1_version3::launch_flash_attention(d_Q, d_K, d_V, d_O, d_l, d_m,
                                                 batch, heads, N, d, stream);
        },
        2e-2f, 2e-2f);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ Attention Benchmark ================\n";
    std::cout << "batch=" << batch << ", heads=" << heads
              << ", N=" << N << ", d=" << d << "\n";
    std::cout << "warmup_iters=" << warmup__iters
              << ", bench_iters=" << bench__iters << "\n";
    std::cout << "(fa1_version1/2/3 的计时不包含每轮 O/l/m 预初始化时间；预初始化在计时区间外完成)\n\n";

    for (const auto& r : reports) {
        std::cout << std::left << std::setw(16) << r.name
                  << " avg=" << std::setw(10) << r.avg_ms << " ms";
        if (r.name != "baseline") {
            std::cout << " speedup_vs_baseline=" << std::setw(10) << (baseline_ms / r.avg_ms) << "x"
                      << " correct=" << (r.pass ? "PASS" : "FAIL")
                      << " max_abs=" << r.max_abs_err
                      << " max_rel=" << r.max_rel_err;
        }
        std::cout << "\n";
    }
    std::cout << "=====================================================\n";

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_O_ref));
    CHECK_CUDA(cudaFree(d_O));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_P));
    CHECK_CUDA(cudaFree(d_l));
    CHECK_CUDA(cudaFree(d_m));
    return 0;
}
