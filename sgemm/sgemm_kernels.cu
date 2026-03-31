#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mma.h>

namespace Naive {
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(32, 32);
    dim3 gridDim(N / 32, M / 32);
    sgemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}

namespace Block_Tile {
template <int BM, int BN, int BK>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    float sum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        for (int i = tx; i < BK; i += blockDim.x) {
            s_A[ty][i] = A[offset_A + ty * K + (k + i)];
        }
        for (int i = ty; i < BK; i += blockDim.y) {
            s_B[i][tx] = B[offset_B + (k + i) * N + tx];
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < BK; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }
        __syncthreads();
    }
    C[offset_C + ty * N + tx] = sum;
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    dim3 blockDim(BN, BM);
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}

namespace Thread_Tile {
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;
    const int trow = threadIdx.x / (BN / TN);
    const int tcol = threadIdx.x % (BN / TN);
    const int ty = trow * TM;
    const int tx = tcol * TN;

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    __shared__ float s_A[BM][BK];
    __shared__ float s_B[BK][BN];

    float temp[TM][TN] = { 0.0f };
    float reg_A[TM];
    float reg_B[TN];

    for (int k = 0; k < K; k += BK) {
        for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
            int row = i / BK;
            int col = i % BK;
            s_A[row][col] = A[offset_A + row * K + (k + col)];
        }
        for (int i = threadIdx.x; i < BK * BN; i += blockDim.x) {
            int row = i / BN;
            int col = i % BN;
            s_B[row][col] = B[offset_B + (k + row) * N + col];
        }
        __syncthreads();

#pragma unroll
        for (int x = 0; x < BK; ++x) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[ty + i][x];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[x][tx + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    temp[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            C[offset_C + (ty + i) * N + (tx + j)] = temp[i][j];
        }
    }
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 blockDim((BN / TN) * (BM / TM));
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}

namespace Vectorized_LDST {
template <int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;
    const int trow = threadIdx.x / (BN / TN);
    const int tcol = threadIdx.x % (BN / TN);
    const int ty = trow * TM;
    const int tx = tcol * TN;

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    // 通过 padding 避免 bank conflict
    constexpr int pad = 4;
    __shared__ float s_A[BK * (BM + pad)];
    __shared__ float s_B[BK * BN];

    float temp[TM][TN] = { 0.0f };
    float reg_A[TM];
    float reg_B[TN];

    for (int k = 0; k < K; k += BK) {
        for (int i = (threadIdx.x << 2); i < BM * BK; i += (blockDim.x << 2)) {
            int row = i / BK;
            int col = i % BK;
            float4 v4 = reinterpret_cast<const float4*>(A + offset_A + row * K + (k + col))[0];
            s_A[col * (BM + pad) + row] = v4.x;
            s_A[(col + 1) * (BM + pad) + row] = v4.y;
            s_A[(col + 2) * (BM + pad) + row] = v4.z;
            s_A[(col + 3) * (BM + pad) + row] = v4.w;
        }
        for (int i = (threadIdx.x << 2); i < BK * BN; i += (blockDim.x << 2)) {
            int row = i / BN;
            int col = i % BN;
            reinterpret_cast<float4*>(s_B + row * BN + col)[0] = 
            reinterpret_cast<const float4*>(B + offset_B + (k + row) * N + col)[0];
        }
        __syncthreads();

#pragma unroll
        for (int x = 0; x < BK; ++x) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[i] = s_A[x * (BM + pad) + ty + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[j] = s_B[x * BN + tx + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    temp[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; j += 4) {
            reinterpret_cast<float4*>(C + offset_C + (ty + i) * N + (tx + j))[0] = 
            make_float4(temp[i][j], temp[i][j + 1], temp[i][j + 2], temp[i][j + 3]);
        }
    }
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 blockDim((BN / TN) * (BM / TM));
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}

namespace Warp_Tile {
template <int BM, int BN, int BK, int PAD>
__device__ void load_from_gmem_to_smem(
    const float* A, const float* B, 
    float* s_A, float* s_B, 
    int offset_A, int offset_B, 
    int N, int K, int k
) {
    int row_a, col_a, row_b, col_b;
    float4 v4;
    for (int i = (threadIdx.x << 2); i < BM * BK; i += (blockDim.x << 2)) {
        row_a = i / BK;
        col_a = i % BK;
        v4 = reinterpret_cast<const float4*>(A + offset_A + row_a * K + (k + col_a))[0];
        s_A[col_a * (BM + PAD) + row_a] = v4.x;
        s_A[(col_a + 1) * (BM + PAD) + row_a] = v4.y;
        s_A[(col_a + 2) * (BM + PAD) + row_a] = v4.z;
        s_A[(col_a + 3) * (BM + PAD) + row_a] = v4.w;
    }
    for (int i = (threadIdx.x << 2); i < BK * BN; i += (blockDim.x << 2)) {
        row_b = i / BN;
        col_b = i % BN;
        reinterpret_cast<float4*>(s_B + row_b * BN + col_b)[0] = 
        reinterpret_cast<const float4*>(B + offset_B + (k + row_b) * N + col_b)[0];
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WM_SUB, int WN_SUB, int TM, int TN, int WM_ITERS, int WN_ITERS, int PAD>
__device__ void gemm_from_smem_to_reg(
    float* s_A, float* s_B, 
    float* reg_A, float* reg_B, float* temp, 
    int warp_row, int warp_col, 
    int trow, int tcol
) {
#pragma unroll
    for (int x = 0; x < BK; ++x) {
#pragma unroll
        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[wm_idx * TM + i] = s_A[x * (BM + PAD) + warp_row * WM + wm_idx * WM_SUB + trow * TM + i];
            }
        }
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[wn_idx * TN + j] = s_B[x * BN + warp_col * WN + wn_idx * WN_SUB + tcol * TN + j];
            }
        }
#pragma unroll
        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
            for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + wn_idx * TN + j] 
                        += reg_A[wm_idx * TM + i] * reg_B[wn_idx * TN + j];
                    }
                }
            }
        }
    }
}

template <int WM, int WN, int WM_SUB, int WN_SUB, int TM, int TN, int WM_ITERS, int WN_ITERS>
__device__ void store_from_reg_to_gmem(
    float* C, float* temp, 
    int offset_C, 
    int warp_row, int warp_col, 
    int trow, int tcol, 
    int N
) {
    int offset = offset_C + (warp_row * WM + trow * TM) * N + (warp_col * WN + tcol * TN);
#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    reinterpret_cast<float4*>(C + offset + (wm_idx * WM_SUB + i) * N + (wn_idx * WN_SUB + j))[0] = make_float4(
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j)], 
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 1)], 
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 2)], 
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 3)]
                    );
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;

    // block 内 warp 二维分布的 id
    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    // M、N 方向每个 warp 迭代次数 (2, 2)
    constexpr int WM_ITERS = 2;
    constexpr int WN_ITERS = (WM * WN) / (TM * TN * 32 * WM_ITERS);

    // warp 每次迭代处理的 M、N 方向上的元素个数 (32, 16)
    constexpr int WM_SUB = WM / WM_ITERS;
    constexpr int WN_SUB = WN / WN_ITERS;

    // warp 内 thread 二维分布的 id: (0..8, 0..4)
    const int lane = threadIdx.x & 31;
    const int trow = lane / (WN_SUB / TN);
    const int tcol = lane % (WN_SUB / TN);

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    // 通过 padding 避免 bank conflict
    constexpr int PAD = 4;
    __shared__ float s_A[BK * (BM + PAD)];
    __shared__ float s_B[BK * BN];

    float temp[(TM * WM_ITERS) * (TN * WN_ITERS)] = { 0.0f };
    float reg_A[TM * WM_ITERS];
    float reg_B[TN * WN_ITERS];

    for (int k = 0; k < K; k += BK) {
        load_from_gmem_to_smem<BM, BN, BK, PAD>(A, B, s_A, s_B, offset_A, offset_B, N, K, k);
        __syncthreads();

        gemm_from_smem_to_reg<BM, BN, BK, WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS, PAD>(
            s_A, s_B, reg_A, reg_B, temp, warp_row, warp_col, trow, tcol
        );
        __syncthreads();
    }
    store_from_reg_to_gmem<WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS>(
        C, temp, offset_C, warp_row, warp_col, trow, tcol, N
    );
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    dim3 blockDim((BN / WN) * (BM / WM) * 32);
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK, WM, WN, TM, TN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}

namespace Double_Buffer {
template <int BM, int BN, int BK, int PAD>
__device__ void load_from_gmem_to_smem(
    const float* A, const float* B, 
    float* s_A, float* s_B, 
    int offset_A, int offset_B, 
    int N, int K, int k, 
    int buffer_id
) {
    int row_a, col_a, row_b, col_b;
    float4 v4;
    for (int i = (threadIdx.x << 2); i < BM * BK; i += (blockDim.x << 2)) {
        row_a = i / BK;
        col_a = i % BK;
        v4 = reinterpret_cast<const float4*>(A + offset_A + row_a * K + (k + col_a))[0];
        s_A[buffer_id * BK * (BM + PAD) + col_a * (BM + PAD) + row_a] = v4.x;
        s_A[buffer_id * BK * (BM + PAD) + (col_a + 1) * (BM + PAD) + row_a] = v4.y;
        s_A[buffer_id * BK * (BM + PAD) + (col_a + 2) * (BM + PAD) + row_a] = v4.z;
        s_A[buffer_id * BK * (BM + PAD) + (col_a + 3) * (BM + PAD) + row_a] = v4.w;
    }
    for (int i = (threadIdx.x << 2); i < BK * BN; i += (blockDim.x << 2)) {
        row_b = i / BN;
        col_b = i % BN;
        reinterpret_cast<float4*>(s_B + buffer_id * BK * BN + row_b * BN + col_b)[0] = 
        reinterpret_cast<const float4*>(B + offset_B + (k + row_b) * N + col_b)[0];
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WM_SUB, int WN_SUB, int TM, int TN, int WM_ITERS, int WN_ITERS, int PAD>
__device__ void gemm_from_smem_to_reg(
    float* s_A, float* s_B, 
    float* reg_A, float* reg_B, float* temp, 
    int warp_row, int warp_col, 
    int trow, int tcol, 
    int buffer_id
) {
#pragma unroll
    for (int x = 0; x < BK; ++x) {
#pragma unroll
        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[wm_idx * TM + i] 
                = s_A[buffer_id * BK * (BM + PAD) + x * (BM + PAD) + warp_row * WM + wm_idx * WM_SUB + trow * TM + i];
            }
        }
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[wn_idx * TN + j] 
                = s_B[buffer_id * BK * BN + x * BN + warp_col * WN + wn_idx * WN_SUB + tcol * TN + j];
            }
        }
#pragma unroll
        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
            for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + wn_idx * TN + j] 
                        += reg_A[wm_idx * TM + i] * reg_B[wn_idx * TN + j];
                    }
                }
            }
        }
    }
}

template <int WM, int WN, int WM_SUB, int WN_SUB, int TM, int TN, int WM_ITERS, int WN_ITERS>
__device__ void store_from_reg_to_gmem(
    float* C, float* temp, 
    int offset_C, 
    int warp_row, int warp_col, 
    int trow, int tcol, 
    int N
) {
    int offset = offset_C + (warp_row * WM + trow * TM) * N + (warp_col * WN + tcol * TN);
#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; j += 4) {
                    reinterpret_cast<float4*>(C + offset + (wm_idx * WM_SUB + i) * N + (wn_idx * WN_SUB + j))[0] = make_float4(
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j)], 
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 1)], 
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 2)], 
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 3)]
                    );
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;

    // block 内 warp 二维分布的 id
    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    // M、N 方向每个 warp 迭代次数 (2, 2)
    constexpr int WM_ITERS = 2;
    constexpr int WN_ITERS = (WM * WN) / (TM * TN * 32 * WM_ITERS);

    // warp 每次迭代处理的 M、N 方向上的元素个数 (32, 16)
    constexpr int WM_SUB = WM / WM_ITERS;
    constexpr int WN_SUB = WN / WN_ITERS;

    // warp 内 thread 二维分布的 id: (0..8, 0..4)
    const int lane = threadIdx.x & 31;
    const int trow = lane / (WN_SUB / TN);
    const int tcol = lane % (WN_SUB / TN);

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    // 通过 padding 避免 bank conflict
    constexpr int PAD = 4;
    __shared__ float s_A[2 * BK * (BM + PAD)];
    __shared__ float s_B[2 * BK * BN];

    float temp[(TM * WM_ITERS) * (TN * WN_ITERS)] = { 0.0f };
    float reg_A[TM * WM_ITERS];
    float reg_B[TN * WN_ITERS];

    int buffer_id = 0;
    load_from_gmem_to_smem<BM, BN, BK, PAD>(A, B, s_A, s_B, offset_A, offset_B, N, K, 0, buffer_id);
    __syncthreads();

    for (int k = 0; k < K - BK; k += BK) {
        load_from_gmem_to_smem<BM, BN, BK, PAD>(A, B, s_A, s_B, offset_A, offset_B, N, K, k + BK, buffer_id ^ 1);

        gemm_from_smem_to_reg<BM, BN, BK, WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS, PAD>(
            s_A, s_B, reg_A, reg_B, temp, warp_row, warp_col, trow, tcol, buffer_id
        );

        buffer_id ^= 1;
        __syncthreads();
    }
    gemm_from_smem_to_reg<BM, BN, BK, WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS, PAD>(
        s_A, s_B, reg_A, reg_B, temp, warp_row, warp_col, trow, tcol, buffer_id
    );

    store_from_reg_to_gmem<WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS>(
        C, temp, offset_C, warp_row, warp_col, trow, tcol, N
    );
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int TM = 4;
    constexpr int TN = 4;
    dim3 blockDim((BN / WN) * (BM / WM) * 32);
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK, WM, WN, TM, TN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}

namespace Tensor_Core_WMMA_HALF {
template <int BM, int BN, int BK>
__device__ void load_from_gmem_to_smem_half(
    const float* A, const float* B, 
    half* s_A, half* s_B, 
    int offset_A, int offset_B, 
    int N, int K, int k, 
    int buffer_id
) {
    int row_a, col_a, row_b, col_b;
    float4 v4;
    for (int i = (threadIdx.x << 2); i < BM * BK; i += (blockDim.x << 2)) {
        row_a = i / BK;
        col_a = i % BK;
        v4 = reinterpret_cast<const float4*>(A + offset_A + row_a * K + (k + col_a))[0];
        s_A[buffer_id * BM * BK + row_a * BK + col_a] = __float2half(v4.x);
        s_A[buffer_id * BM * BK + row_a * BK + col_a + 1] = __float2half(v4.y);
        s_A[buffer_id * BM * BK + row_a * BK + col_a + 2] = __float2half(v4.z);
        s_A[buffer_id * BM * BK + row_a * BK + col_a + 3] = __float2half(v4.w);
    }
    for (int i = (threadIdx.x << 2); i < BK * BN; i += (blockDim.x << 2)) {
        row_b = i / BN;
        col_b = i % BN;
        v4 = reinterpret_cast<const float4*>(B + offset_B + (k + row_b) * N + col_b)[0];
        s_B[buffer_id * BK * BN + row_b * BN + col_b] = __float2half(v4.x);
        s_B[buffer_id * BK * BN + row_b * BN + col_b + 1] = __float2half(v4.y);
        s_B[buffer_id * BK * BN + row_b * BN + col_b + 2] = __float2half(v4.z);
        s_B[buffer_id * BK * BN + row_b * BN + col_b + 3] = __float2half(v4.w);
    }
}

template <int BM, int BN, int BK, int WM, int WN, int WM_ITERS, int WN_ITERS, int WK_ITERS, typename T1, typename T2, typename T3>
__device__ void gemm_from_smem_half_to_frag(
    half* s_A, half* s_B, 
    T1* a_frag, T2* b_frag, T3* c_frag, 
    int warp_row, int warp_col, 
    int buffer_id
) {
    using namespace nvcuda;
    int offset;

#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
            offset = (buffer_id * BM * BK) + (warp_row * WM + wm_idx * 16) * BK + (wk_idx * 16);
            wmma::load_matrix_sync(a_frag[wm_idx * WK_ITERS + wk_idx], s_A + offset, BK);
        }
    }

#pragma unroll
    for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
            offset = (buffer_id * BK * BN) + (wk_idx * 16) * BN + (warp_col * WN + wn_idx * 16);
            wmma::load_matrix_sync(b_frag[wk_idx * WN_ITERS + wn_idx], s_B + offset, BN);
        }
    }

#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
                wmma::mma_sync(
                    c_frag[wm_idx * WN_ITERS + wn_idx], 
                    a_frag[wm_idx * WK_ITERS + wk_idx], 
                    b_frag[wk_idx * WN_ITERS + wn_idx], 
                    c_frag[wm_idx * WN_ITERS + wn_idx]
                );
            }
        }
    }
}

template <int WM, int WN, int WM_ITERS, int WN_ITERS, typename T>
__device__ void store_from_frag_to_gmem(float* C, int offset_C, T* c_frag, int warp_row, int warp_col, int N) {
    using namespace nvcuda;
    int offset;

#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
            offset = offset_C + (warp_row * WM + wm_idx * 16) * N + (warp_col * WN + wn_idx * 16);
            wmma::store_matrix_sync(C + offset, c_frag[wm_idx * WN_ITERS + wn_idx], N, wmma::mem_row_major);
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    using namespace nvcuda;

    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;

    // block 内 warp 二维分布的 id
    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    // M、N、K 方向每个 warp 迭代次数
    constexpr int WM_ITERS = WM / 16;
    constexpr int WN_ITERS = WN / 16;
    constexpr int WK_ITERS = BK / 16;

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    __shared__ half s_A[2 * BM * BK];
    __shared__ half s_B[2 * BK * BN];

    using FragAType = wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major>;
    using FragCType = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

    FragAType a_frag[WM_ITERS * WK_ITERS];
    FragBType b_frag[WK_ITERS * WN_ITERS];
    FragCType c_frag[WM_ITERS * WN_ITERS];

#pragma unroll
    for (int i = 0; i < WM_ITERS * WN_ITERS; ++i) {
        wmma::fill_fragment(c_frag[i], 0.0f);
    }

    int buffer_id = 0;
    load_from_gmem_to_smem_half<BM, BN, BK>(A, B, s_A, s_B, offset_A, offset_B, N, K, 0, buffer_id);
    __syncthreads();

    for (int k = 0; k < K - BK; k += BK) {
        load_from_gmem_to_smem_half<BM, BN, BK>(A, B, s_A, s_B, offset_A, offset_B, N, K, k + BK, buffer_id ^ 1);

        gemm_from_smem_half_to_frag<BM, BN, BK, WM, WN, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragBType, FragCType>(
            s_A, s_B, a_frag, b_frag, c_frag, warp_row, warp_col, buffer_id
        );

        buffer_id ^= 1;
        __syncthreads();
    }
    gemm_from_smem_half_to_frag<BM, BN, BK, WM, WN, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragBType, FragCType>(
        s_A, s_B, a_frag, b_frag, c_frag, warp_row, warp_col, buffer_id
    );

    store_from_frag_to_gmem<WM, WN, WM_ITERS, WN_ITERS, FragCType>(
        C, offset_C, c_frag, warp_row, warp_col, N
    );
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 32;
    dim3 blockDim((BN / WN) * (BM / WM) * 32);
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK, WM, WN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}