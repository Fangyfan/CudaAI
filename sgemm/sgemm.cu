#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__            \
                      << std::endl;                                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

#define CHECK_CUBLAS(call)                                                     \
    do {                                                                       \
        cublasStatus_t status__ = (call);                                      \
        if (status__ != CUBLAS_STATUS_SUCCESS) {                               \
            std::cerr << "cuBLAS Error: " << static_cast<int>(status__)      \
                      << " at " << __FILE__ << ":" << __LINE__            \
                      << std::endl;                                             \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                      \
    } while (0)

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
}  // namespace Naive

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
}  // namespace Block_Tile

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

    float temp[TM][TN] = {0.0f};
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
}  // namespace Thread_Tile

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

    constexpr int pad = 4;
    __shared__ float s_A[BK * (BM + pad)];
    __shared__ float s_B[BK * BN];

    float temp[TM][TN] = {0.0f};
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
}  // namespace Vectorized_LDST

namespace Warp_Tile {
template <int BM, int BN, int BK, int PAD>
__device__ void load_from_gmem_to_smem(
    const float* A, const float* B,
    float* s_A, float* s_B,
    int offset_A, int offset_B,
    int N, int K, int k) {
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
    int trow, int tcol) {
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
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + wn_idx * TN + j] +=
                            reg_A[wm_idx * TM + i] * reg_B[wn_idx * TN + j];
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
    int N) {
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
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 3)]);
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;

    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    constexpr int WM_ITERS = 2;
    constexpr int WN_ITERS = (WM * WN) / (TM * TN * 32 * WM_ITERS);
    constexpr int WM_SUB = WM / WM_ITERS;
    constexpr int WN_SUB = WN / WN_ITERS;

    const int lane = threadIdx.x & 31;
    const int trow = lane / (WN_SUB / TN);
    const int tcol = lane % (WN_SUB / TN);

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    constexpr int PAD = 4;
    __shared__ float s_A[BK * (BM + PAD)];
    __shared__ float s_B[BK * BN];

    float temp[(TM * WM_ITERS) * (TN * WN_ITERS)] = {0.0f};
    float reg_A[TM * WM_ITERS];
    float reg_B[TN * WN_ITERS];

    for (int k = 0; k < K; k += BK) {
        load_from_gmem_to_smem<BM, BN, BK, PAD>(A, B, s_A, s_B, offset_A, offset_B, N, K, k);
        __syncthreads();

        gemm_from_smem_to_reg<BM, BN, BK, WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS, PAD>(
            s_A, s_B, reg_A, reg_B, temp, warp_row, warp_col, trow, tcol);
        __syncthreads();
    }
    store_from_reg_to_gmem<WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS>(
        C, temp, offset_C, warp_row, warp_col, trow, tcol, N);
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
}  // namespace Warp_Tile

namespace Double_Buffer {
template <int BM, int BN, int BK, int PAD>
__device__ void load_from_gmem_to_smem(
    const float* A, const float* B,
    float* s_A, float* s_B,
    int offset_A, int offset_B,
    int N, int K, int k,
    int buffer_id) {
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
    int buffer_id) {
#pragma unroll
    for (int x = 0; x < BK; ++x) {
#pragma unroll
        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_A[wm_idx * TM + i] =
                    s_A[buffer_id * BK * (BM + PAD) + x * (BM + PAD) + warp_row * WM + wm_idx * WM_SUB + trow * TM + i];
            }
        }
#pragma unroll
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_B[wn_idx * TN + j] =
                    s_B[buffer_id * BK * BN + x * BN + warp_col * WN + wn_idx * WN_SUB + tcol * TN + j];
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
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + wn_idx * TN + j] +=
                            reg_A[wm_idx * TM + i] * reg_B[wn_idx * TN + j];
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
    int N) {
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
                        temp[(wm_idx * TM + i) * (WN_ITERS * TN) + (wn_idx * TN + j + 3)]);
                }
            }
        }
    }
}

template <int BM, int BN, int BK, int WM, int WN, int TM, int TN>
__global__ void sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    const int by = blockIdx.y * BM;
    const int bx = blockIdx.x * BN;

    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    constexpr int WM_ITERS = 2;
    constexpr int WN_ITERS = (WM * WN) / (TM * TN * 32 * WM_ITERS);
    constexpr int WM_SUB = WM / WM_ITERS;
    constexpr int WN_SUB = WN / WN_ITERS;

    const int lane = threadIdx.x & 31;
    const int trow = lane / (WN_SUB / TN);
    const int tcol = lane % (WN_SUB / TN);

    const int offset_A = by * K;
    const int offset_B = bx;
    const int offset_C = by * N + bx;

    constexpr int PAD = 4;
    __shared__ float s_A[2 * BK * (BM + PAD)];
    __shared__ float s_B[2 * BK * BN];

    float temp[(TM * WM_ITERS) * (TN * WN_ITERS)] = {0.0f};
    float reg_A[TM * WM_ITERS];
    float reg_B[TN * WN_ITERS];

    int buffer_id = 0;
    load_from_gmem_to_smem<BM, BN, BK, PAD>(A, B, s_A, s_B, offset_A, offset_B, N, K, 0, buffer_id);
    __syncthreads();

    for (int k = 0; k < K - BK; k += BK) {
        load_from_gmem_to_smem<BM, BN, BK, PAD>(A, B, s_A, s_B, offset_A, offset_B, N, K, k + BK, buffer_id ^ 1);

        gemm_from_smem_to_reg<BM, BN, BK, WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS, PAD>(
            s_A, s_B, reg_A, reg_B, temp, warp_row, warp_col, trow, tcol, buffer_id);

        buffer_id ^= 1;
        __syncthreads();
    }
    gemm_from_smem_to_reg<BM, BN, BK, WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS, PAD>(
        s_A, s_B, reg_A, reg_B, temp, warp_row, warp_col, trow, tcol, buffer_id);

    store_from_reg_to_gmem<WM, WN, WM_SUB, WN_SUB, TM, TN, WM_ITERS, WN_ITERS>(
        C, temp, offset_C, warp_row, warp_col, trow, tcol, N);
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
}  // namespace Double_Buffer

namespace Tensor_Core_WMMA_HALF {
template <int BM, int BN, int BK>
__device__ void load_from_gmem_to_smem_half(
    const float* A, const float* B,
    half* s_A, half* s_B,
    int offset_A, int offset_B,
    int N, int K, int k,
    int buffer_id) {
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
    int buffer_id) {
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
                    c_frag[wm_idx * WN_ITERS + wn_idx]);
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

    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

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
            s_A, s_B, a_frag, b_frag, c_frag, warp_row, warp_col, buffer_id);

        buffer_id ^= 1;
        __syncthreads();
    }
    gemm_from_smem_half_to_frag<BM, BN, BK, WM, WN, WM_ITERS, WN_ITERS, WK_ITERS, FragAType, FragBType, FragCType>(
        s_A, s_B, a_frag, b_frag, c_frag, warp_row, warp_col, buffer_id);

    store_from_frag_to_gmem<WM, WN, WM_ITERS, WN_ITERS, FragCType>(
        C, offset_C, c_frag, warp_row, warp_col, N);
}

void launch_sgemm_kernel(const float* A, const float* B, float* C, int M, int N, int K, cudaStream_t stream) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int WM = 32;
    constexpr int WN = 32;
    dim3 blockDim((BN / WN) * (BM / WM) * 32);
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<BM, BN, BK, WM, WN><<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}
}  // namespace Tensor_Core_WMMA_HALF


namespace Tensor_Core_WMMA_TF32 {
#define BLOCK_DIM 512
#define WARPS_Y 4
#define WARPS_X 4
#define WM_ITERS 2
#define WN_ITERS 2
#define WK_ITERS 2
#define TC_M 16
#define TC_N 16
#define TC_K 8
#define BM 128  // (WARPS_Y * WM_ITERS * TC_M)
#define BN 128  // (WARPS_X * WN_ITERS * TC_N)
#define BK 16   // (WK_ITERS * TC_K)
#define WM 32   // (WM_ITERS * TC_M)
#define WN 32   // (WN_ITERS * TC_N)
#define PAD 8
#define FLOAT2(addr) *(reinterpret_cast<float2*>(addr))
#define FLOAT4(addr) *(reinterpret_cast<float4*>(addr))

#define CP_ASYNC_CA_SHARED_GLOBAL_FLOAT2(saddr, gaddr) asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"((unsigned)__cvta_generic_to_shared(saddr)), "l"(gaddr))
#define CP_ASYNC_CA_SHARED_GLOBAL_FLOAT4(saddr, gaddr) asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" :: "r"((unsigned)__cvta_generic_to_shared(saddr)), "l"(gaddr))
#define CP_ASYNC_COMMIT_GROUP asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP_0 asm volatile("cp.async.wait_group 0;\n" ::)
#define CP_ASYNC_WAIT_GROUP_1 asm volatile("cp.async.wait_group 1;\n" ::)

__device__ __forceinline__ void load_from_gmem_to_smem_A(float* s_A, const float* A, int K, int k) {
    constexpr int load_nfloats_per_thread = BM * BK / BLOCK_DIM;
    constexpr int nfloat2_per_row = BK / load_nfloats_per_thread;
    int trow = threadIdx.x / nfloat2_per_row;
    int tcol = threadIdx.x % nfloat2_per_row;
    int grow = (blockIdx.y * BM) + trow;
    int gcol = k + (tcol * load_nfloats_per_thread);
    CP_ASYNC_CA_SHARED_GLOBAL_FLOAT4(s_A + load_nfloats_per_thread * threadIdx.x,
                                     A + grow * K + gcol);
}

__device__ __forceinline__ void load_from_gmem_to_smem_B(float* s_B, const float* B, int N, int k) {
    constexpr int load_nfloats_per_thread = BK * BN / BLOCK_DIM;
    constexpr int nfloat4_per_row = BN / load_nfloats_per_thread;
    int trow = threadIdx.x / nfloat4_per_row;
    int tcol = threadIdx.x % nfloat4_per_row;
    int grow = k + trow;
    int gcol = (blockIdx.x * BN) + (tcol * load_nfloats_per_thread);
    CP_ASYNC_CA_SHARED_GLOBAL_FLOAT4(s_B + trow * (BN + PAD) + tcol * load_nfloats_per_thread,
                                     B + grow * N + gcol);
}

__global__ void sgemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K) {
    using namespace nvcuda;

    const int warp_id = threadIdx.x >> 5;
    const int warp_row = warp_id / WARPS_X;
    const int warp_col = warp_id % WARPS_X;

    __shared__ float s_A[2][BM * BK];
    __shared__ float s_B[2][BK * (BN + PAD)];

    using FragAType = wmma::fragment<wmma::matrix_a, TC_M, TC_N, TC_K, wmma::precision::tf32, wmma::row_major>;
    using FragBType = wmma::fragment<wmma::matrix_b, TC_M, TC_N, TC_K, wmma::precision::tf32, wmma::row_major>;
    using FragCType = wmma::fragment<wmma::accumulator, TC_M, TC_N, TC_K, float>;

    FragAType a_frag[WM_ITERS][WK_ITERS];
    FragBType b_frag[WN_ITERS][WK_ITERS];
    FragCType c_frag[WM_ITERS][WN_ITERS];

    for (int i = 0; i < WM_ITERS; ++i) {
        for (int j = 0; j < WN_ITERS; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    int buffer_id = 0;
    load_from_gmem_to_smem_A(s_A[buffer_id], A, K, 0);
    load_from_gmem_to_smem_B(s_B[buffer_id], B, N, 0);
    CP_ASYNC_COMMIT_GROUP;
    CP_ASYNC_WAIT_GROUP_0;
    __syncthreads();

    for (int k = 0; k < K; k += BK) {
        if (k + BK < K) {
            load_from_gmem_to_smem_A(s_A[buffer_id ^ 1], A, K, k + BK);
            load_from_gmem_to_smem_B(s_B[buffer_id ^ 1], B, N, k + BK);
            CP_ASYNC_COMMIT_GROUP;
        }

        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
            for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
                int offset = (warp_row * WM + wm_idx * TC_M) * BK + (wk_idx * TC_K);
                wmma::load_matrix_sync(a_frag[wm_idx][wk_idx], s_A[buffer_id] + offset, BK);
            }
        }

        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
            for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
                int offset = (wk_idx * TC_K) * (BN + PAD) + (warp_col * WN + wn_idx * TC_N);
                wmma::load_matrix_sync(b_frag[wn_idx][wk_idx], s_B[buffer_id] + offset, BN + PAD);
            }
        }

        for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
            for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
                for (int wk_idx = 0; wk_idx < WK_ITERS; ++wk_idx) {
                    wmma::mma_sync(c_frag[wm_idx][wn_idx],
                                   a_frag[wm_idx][wk_idx],
                                   b_frag[wn_idx][wk_idx],
                                   c_frag[wm_idx][wn_idx]);
                }
            }
        }

        buffer_id ^= 1;
        CP_ASYNC_WAIT_GROUP_0;
        __syncthreads();
    }

    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
        for (int wn_idx = 0; wn_idx < WN_ITERS; ++wn_idx) {
            int row = (blockIdx.y * BM) + (warp_row * WM) + (wm_idx * TC_M);
            int col = (blockIdx.x * BN) + (warp_col * WN) + (wn_idx * TC_N);
            wmma::store_matrix_sync(C + row * N + col,
                                    c_frag[wm_idx][wn_idx],
                                    N,
                                    wmma::mem_row_major);
        }
    }
}

void launch_sgemm_kernel(const float* A, const float* B, float* C,
                         int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(N / BN, M / BM);
    sgemm_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, M, N, K);
}

#undef BLOCK_DIM
#undef WARPS_Y
#undef WARPS_X
#undef WM_ITERS
#undef WN_ITERS
#undef WK_ITERS
#undef TC_M
#undef TC_N
#undef TC_K
#undef BM
#undef BN
#undef BK
#undef WM
#undef WN
#undef PAD
#undef FLOAT2
#undef FLOAT4
#undef CP_ASYNC_CA_SHARED_GLOBAL_FLOAT2
#undef CP_ASYNC_CA_SHARED_GLOBAL_FLOAT4
#undef CP_ASYNC_COMMIT_GROUP
#undef CP_ASYNC_WAIT_GROUP_0
#undef CP_ASYNC_WAIT_GROUP_1
}  // namespace Tensor_Core_WMMA_TF32

namespace baseline {
void launch_cublas_sgemm(const float* A, const float* B, float* C,
                         int M, int N, int K,
                         cublasHandle_t handle, cudaStream_t stream) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CHECK_CUBLAS(cublasSetStream(handle, stream));
    // Row-major C = A * B can be computed via column-major cuBLAS as:
    // C^T = B^T * A^T, which maps to cublasSgemm with (N, M, K).
    CHECK_CUBLAS(cublasSgemm(handle,
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}
}  // namespace baseline

struct ErrorStats {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    size_t max_abs_idx = 0;
    size_t max_rel_idx = 0;
    bool pass = true;
};

ErrorStats compare_results(const std::vector<float>& ref,
                           const std::vector<float>& test,
                           float abs_tol,
                           float rel_tol) {
    ErrorStats stats;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(ref[i] - test[i]);
        float rel = diff / (std::fabs(ref[i]) + 1e-6f);
        if (diff > stats.max_abs) {
            stats.max_abs = diff;
            stats.max_abs_idx = i;
        }
        if (rel > stats.max_rel) {
            stats.max_rel = rel;
            stats.max_rel_idx = i;
        }
        if (!(diff <= abs_tol + rel_tol * std::fabs(ref[i]))) {
            stats.pass = false;
        }
    }
    return stats;
}

template <typename LaunchFn>
float benchmark_kernel(LaunchFn launch,
                       int warmup_iters,
                       int bench_iters,
                       cudaStream_t stream) {
    for (int i = 0; i < warmup_iters; ++i) {
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start, stream));
    for (int i = 0; i < bench_iters; ++i) {
        launch();
    }
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaGetLastError());

    float total_ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&total_ms, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / bench_iters;
}

inline double calc_tflops(int M, int N, int K, float ms) {
    return (2.0 * static_cast<double>(M) * N * K) / (static_cast<double>(ms) * 1.0e9);
}

struct KernelResult {
    std::string name;
    float avg_ms;
    double tflops;
    ErrorStats err;
};

int main() {
    constexpr int M = 4096;
    constexpr int N = 4096;
    constexpr int K = 4096;
    constexpr int warmup_iters = 20;
    constexpr int bench_iters = 50;

    const size_t size_A = static_cast<size_t>(M) * K;
    const size_t size_B = static_cast<size_t>(K) * N;
    const size_t size_C = static_cast<size_t>(M) * N;
    const size_t bytes_A = size_A * sizeof(float);
    const size_t bytes_B = size_B * sizeof(float);
    const size_t bytes_C = size_C * sizeof(float);

    std::vector<float> h_A(size_A);
    std::vector<float> h_B(size_B);
    std::vector<float> h_C_ref(size_C);
    std::vector<float> h_C_out(size_C);

    for (size_t i = 0; i < size_A; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (size_t i = 0; i < size_B; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    CHECK_CUDA(cudaSetDevice(3));

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes_B, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetStream(handle, stream));

    auto run_and_collect = [&](const std::string& name,
                               auto launch_fn,
                               float abs_tol,
                               float rel_tol) -> KernelResult {
        auto launch = [&]() {
            launch_fn();
        };
        float avg_ms = benchmark_kernel(launch, warmup_iters, bench_iters, stream);
        CHECK_CUDA(cudaMemcpy(h_C_out.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
        ErrorStats err = compare_results(h_C_ref, h_C_out, abs_tol, rel_tol);
        return KernelResult{name, avg_ms, calc_tflops(M, N, K, avg_ms), err};
    };

    // Baseline
    auto baseline_launch = [&]() {
        baseline::launch_cublas_sgemm(d_A, d_B, d_C, M, N, K, handle, stream);
    };
    float baseline_ms = benchmark_kernel(baseline_launch, warmup_iters, bench_iters, stream);
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C, bytes_C, cudaMemcpyDeviceToHost));
    const double baseline_tflops = calc_tflops(M, N, K, baseline_ms);

    std::vector<KernelResult> results;
    results.push_back(run_and_collect("Naive", [&]() {
        Naive::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-2f, 5e-4f));

    results.push_back(run_and_collect("Block_Tile", [&]() {
        Block_Tile::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-2f, 5e-4f));

    results.push_back(run_and_collect("Thread_Tile", [&]() {
        Thread_Tile::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-2f, 5e-4f));

    results.push_back(run_and_collect("Vectorized_LDST", [&]() {
        Vectorized_LDST::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-2f, 5e-4f));

    results.push_back(run_and_collect("Warp_Tile", [&]() {
        Warp_Tile::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-2f, 5e-4f));

    results.push_back(run_and_collect("Double_Buffer", [&]() {
        Double_Buffer::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-2f, 5e-4f));

    results.push_back(run_and_collect("Tensor_Core_WMMA_HALF", [&]() {
        Tensor_Core_WMMA_HALF::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 5e-1f, 5e-2f));

    results.push_back(run_and_collect("Tensor_Core_WMMA_TF32", [&]() {
        Tensor_Core_WMMA_TF32::launch_sgemm_kernel(d_A, d_B, d_C, M, N, K, stream);
    }, 2e-1f, 2e-3f));

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "M=N=K=" << M << "\n";
    std::cout << "warmup_iters=" << warmup_iters << ", bench_iters=" << bench_iters << "\n\n";
    std::cout << "Baseline (cuBLAS SGEMM): avg_ms=" << baseline_ms
              << ", TFLOPS=" << baseline_tflops << "\n\n";

    std::cout << std::left
              << std::setw(24) << "Kernel"
              << std::setw(12) << "avg_ms"
              << std::setw(12) << "TFLOPS"
              << std::setw(12) << "vs_cublas"
              << std::setw(10) << "Result"
              << std::setw(12) << "max_abs"
              << std::setw(12) << "max_rel"
              << "\n";

    for (const auto& r : results) {
        std::cout << std::left
                  << std::setw(24) << r.name
                  << std::setw(12) << r.avg_ms
                  << std::setw(12) << r.tflops
                  << std::setw(12) << (baseline_ms / r.avg_ms)
                  << std::setw(10) << (r.err.pass ? "PASS" : "FAIL")
                  << std::setw(12) << r.err.max_abs
                  << std::setw(12) << r.err.max_rel
                  << "\n";
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    return 0;
}
