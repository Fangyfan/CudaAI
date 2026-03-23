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


namespace baseline {
template <int BLOCK_DIM>
__global__ void softmax_kernel(const float* input, float* output, int size, float scale) {
    input += blockIdx.x * size;
    output += blockIdx.x * size;

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_val;

    float max = -INFINITY;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        max = fmaxf(max, scale * input[i]);
    }
    max = BlockReduce(temp_storage).Reduce(max, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max;
    }
    __syncthreads();
    max = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        sum += __expf(scale * input[i] - max);
    }
    sum = BlockReduce(temp_storage).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        output[i] = __expf(scale * input[i] - max) / sum;
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


namespace fa1_version3 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
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

__device__ __forceinline__ MD warp_reduce_md(MD val) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = MD_OP()(val, shfl_xor_val(val, delta));
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
#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
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
#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
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
#pragma unroll
    for (int wm_idx = 0; wm_idx < WM_ITERS; ++wm_idx) {
#pragma unroll
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
template <int BLOCK_DIM, int Br, int Bc, int Bd, int Wr, int Wc, int WM_ITERS, int WN_ITERS, int WK_ITERS>
__global__ void __launch_bounds__(BLOCK_DIM) flash_attention_v1_kernel(float* Q, float* K, float* V, float* O, float* l, float* m, int N, int d, float scale) {
    using namespace nvcuda; // wmma

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
                    row_ml_temp = MD_OP()(row_ml_temp, ml_val);
                }
                __syncwarp();

                // 得到 s_S[Br, Bc] 每一行的 m 和 l
                row_ml_temp = warp_reduce_md(row_ml_temp);
                if (lane == 0) {
                    row_ml[r] = row_ml_temp;
                    row_ml_new[r] = MD_OP()(row_ml_old[r], row_ml_temp);
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

void launch_flash_attention_v1(float* Q, float* K, float* V, float* O, float* l, float* m, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    // 让 Bd = Bc 从而使得 S = QK^T 矩阵分片 [Br, Bc] 与 Q / O 矩阵分片 [Br, Bd] 形状相同，方便排布
    constexpr int Br = 32;
    constexpr int Bc = 64;
    constexpr int Bd = Bc;
    constexpr int Wr = 16;
    constexpr int Wc = 16;
    constexpr int BLOCK_DIM = (Br / Wr) * (Bc / Wc) * 32;
    // 单个 warp 处理矩阵乘法 [M,K] × [K,N] = [M,N] 层面 M、N、K 方向每个 warp 的迭代次数
    constexpr int WM_ITERS = Wr / 16;
    constexpr int WN_ITERS = Wc / 16;
    constexpr int WK_ITERS = Bd / 16;
    dim3 gridDim(heads, batch);
    dim3 blockDim(BLOCK_DIM);
    flash_attention_v1_kernel<BLOCK_DIM, Br, Bc, Bd, Wr, Wc, WM_ITERS, WN_ITERS, WK_ITERS><<<gridDim, blockDim, 0, stream>>>(Q, K, V, O, l, m, N, d, scale);
}
}  // namespace fa1_version3


namespace fa2_version1 {
template <int Br, int Bc>
__global__ void flash_attention_v2_kernel(float* Q, float* K, float* V, float* O, int N, int d, float scale) {
    // __shared__ float s_Q[Br][d], s_K[Bc][d + 1], s_V[Bc][d + 1], s_O[Br][d];
    // __shared__ float s_S[Br][Bc], Exp[Br][Bc];
    // __shared__ float Max[Br], SumExp[Br];
    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + Br * d;
    float* s_V = s_K + Bc * (d + 1);
    float* s_O = s_V + Bc * (d + 1);
    float* s_S = s_O + Br * d;
    float* Exp = s_S + Br * Bc;
    float* Max = Exp + Br * Bc;
    float* SumExp = Max + Br;
    
    int tx = threadIdx.x;  // [0, Bc - 1]
    int ty = threadIdx.y;  // [0, Br - 1]
    int offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;
    int r = blockIdx.x * Br + ty;  // 当前 thread 负责计算全局 (Q + offset)[N][d] 的第 r 行

    // block 内的线程组织为 (Br, Bc)
    // 遍历每个块: Q[Br, Bc] + ... + Q[Br, Bc] = Q[Br, d]
    // 每个 thread 搬运自己负责的元素 Q[ty][tx], Q[ty][tx + Bc], Q[ty][tx + 2Bc] , ...
    for (int i = 0; i < d / Bc; ++i) {
        s_Q[ty * d + (tx + i * Bc)] = Q[offset + r * d + (tx + i * Bc)];
        s_O[ty * d + (tx + i * Bc)] = 0.0f;
    }
    if (tx == 0) {
        Max[ty] = -INFINITY;
        SumExp[ty] = 0.0f;
    }
    __syncthreads();

    // 每个 block 负责一个 query 的分块 Q[Br, d]
    // 需要遍历第 j = [0, Tc-1] 个 key / value 分块 K[Bc, d], V[Bc, d]
    for (int j = 0; j < N / Bc; ++j) {
        // block 内的线程组织为 (Br, Bc)
        // 遍历每个块: (K/V)[Bc, Br] + ... + (K/V)[Bc, Br] = (K/V)[Bc, d]
        // 每个 thread 搬运自己负责的 (K/V)[tx][ty], (K/V)[tx][ty + Br], (K/V)[tx][ty + 2Br] , ...
        for (int i = 0; i < d / Br; ++i) {
            s_K[tx * (d + 1) + (ty + i * Br)] = K[offset + (tx + j * Bc) * d + (ty + i * Br)];
            s_V[tx * (d + 1) + (ty + i * Br)] = V[offset + (tx + j * Bc) * d + (ty + i * Br)];
        }
        __syncthreads();

        // 计算点积 S[ty][tx] = QK^T[ty][tx] = Q[ty][0...d-1] * K[tx][0...d-1]
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += s_Q[ty * d + i] * s_K[tx * (d + 1) + i];
        }
        s_S[ty * Bc + tx] = scale * sum; // 注意力分数缩放 QK^T / sqrt(d)
        __syncthreads();

        // 计算当前块 S[Br][Bc] 中当前行 S[ty][...] 的局部最大值 rowMax
        float rowMax = -INFINITY;
        for (int i = 0; i < Bc; ++i) {
            rowMax = fmaxf(rowMax, s_S[ty * Bc + i]);
        }
        // 更新目前的最大值 newMax
        float newMax = fmaxf(Max[ty], rowMax);
        // online softmax 缩放比例: exp(更新前最大值 - 目前最大值)
        float expScale = __expf(Max[ty] - newMax);
        // 计算当前块内指数值 P[ty][tx] = exp(S[ty][tx] - max[ty])
        // 注意这里 max[ty] 是当前 ty 行目前的最大值 newMax
        Exp[ty * Bc + tx] = __expf(s_S[ty * Bc + tx] - newMax);
        __syncthreads();
        
        if (tx == 0) {
            // 计算当前块内 ty 行 Exp[ty][...] 的指数值之和 rowSumExp
            float rowSumExp = 0.0f;
            for (int i = 0; i < Bc; ++i) {
                rowSumExp += Exp[ty * Bc + i];
            }
            // 更新前 SumExp[ty] 是以 [更新前最大值] 作为基准
            // 现在要更新基准为 [目前最大值]，因此乘以 expScale
            // 还要加上当前块内指数值之和 rowSumExp (本来就是以 [目前最大值] 作为基准)
            SumExp[ty] = expScale * SumExp[ty] + rowSumExp;
            Max[ty] = newMax;
        }

        // 计算加权部分和 O_i = expScale * O_i-1 + P_i * V_j
        // block 内的线程组织为 (Br, Bc)
        // 遍历每个块: O[Br, Bc] + ... + O[Br, Bc] = O[Br, d]
        // 每个 thread 负责计算 O[ty][tx], O[ty][tx + Bc], O[ty][tx + 2Bc] , ...
        // O[ty][...] = expScale * O[ty][...] + sum_k(P[ty][k] * V[k][...])
        for (int i = 0; i < d / Bc; ++i) {
            float newO = expScale * s_O[ty * d + (tx + i * Bc)];
            for (int k = 0; k < Bc; ++k) {
                newO += Exp[ty * Bc + k] * s_V[k * (d + 1) + (tx + i * Bc)];
            }
            s_O[ty * d + (tx + i * Bc)] = newO;
        }
        __syncthreads();
    }

    // 每个 thread 负责更新 O[ty][tx], O[ty][tx + Bc], O[ty][tx + 2Bc] , ...
    // 除以 softmax 的分母值 SumExp[ty] = sum(exp(S - max))
    for (int i = 0; i < d / Bc; ++i) {
        O[offset + r * d + (tx + i * Bc)] = s_O[ty * d + (tx + i * Bc)] / SumExp[ty];
    }
}

void launch_flash_attention_v2(float* Q, float* K, float* V, float* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;
    constexpr int Bc = 16;
    int sram_size = (2 * Br * d + 2 * Bc * (d + 1) + 2 * Br * Bc + 2 * Br) * sizeof(float);
    dim3 gridDim(N / Br, heads, batch);
    dim3 blockDim(Bc, Br);
    flash_attention_v2_kernel<Br, Bc><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version1


namespace fa2_version2 {
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, delta);
    }
    return val;
}

template <int Br, int Bc>
__global__ void flash_attention_v2_kernel(float* Q, float* K, float* V, float* O, int N, int d, float scale) {
    // 动态共享内存布局：
    // s_Q    : [Br, d]   当前 block 负责的 Q tile
    // s_K    : [Bc, d]   当前 K tile
    // s_V    : [Bc, d]   当前 V tile
    // s_O    : [Br, d]   输出分子累加器（未除以 SumExp）
    // s_S    : [Br, Bc]  当前 score/prob tile
    // Max    : [Br]      每一行的在线 softmax 最大值
    // SumExp : [Br]      每一行的在线 softmax 分母和
    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + Br * d;
    float* s_V = s_K + Bc * d;
    float* s_O = s_V + Bc * d;
    float* s_S = s_O + Br * d;
    float* Max = s_S + Br * Bc;
    float* SumExp = Max + Br;
    
    // 一个 (batch, head) 对应一张 attention[N, d] 矩阵
    int offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;

    // 当前 block 处理的 Q 起始行
    int r = blockIdx.x * Br;

    // 线程组织：1 warp 负责 1 行 query
    int lane = threadIdx.x & 31;   // warp 内 lane id
    int warp = threadIdx.x >> 5;   // 第几个 warp，对应第几行 query

    // 把当前 block 负责的 Br 行 Q 载入 shared memory
    // 同时把输出分子累加器 s_O 初始化为 0
    for (int i = threadIdx.x; i < Br * d; i += blockDim.x) {
        s_Q[i] = Q[offset + r * d + i];
        s_O[i] = 0.0f;
    }

    // 每个 warp 对应一行，初始化在线 softmax 状态
    if (lane == 0) {
        Max[warp] = -INFINITY;
        SumExp[warp] = 0.0f;
    }
    __syncthreads();

    // 沿 sequence 维分块遍历 K/V
    for (int j = 0; j < N; j += Bc) {
        // 载入当前 K/V tile 到 shared memory
        for (int i = threadIdx.x; i < Bc * d; i += blockDim.x) {
            s_K[i] = K[offset + j * d + i];
            s_V[i] = V[offset + j * d + i];
        }
        __syncthreads();

        // 1) 计算当前 warp (一行 Q) 与当前 K tile 的 score
        //    s_S[warp, col] = scale * dot(Q_row, K_col)
        float rowMax = -INFINITY;
#pragma unroll
        for (int col = 0; col < Bc; ++col) {
            float sum = 0.0f;

            // warp 内并行做一行 Q 和一行 K 的点积
            for (int i = lane; i < d; i += 32) {
                sum += s_Q[warp * d + i] * s_K[col * d + i];
            }
            sum = warp_reduce_sum(sum);

            // lane0 写 score 到 shared memory，供整个 warp 后续使用
            if (lane == 0) {
                s_S[warp * Bc + col] = scale * sum;
            }
            __syncwarp();

            rowMax = fmaxf(rowMax, s_S[warp * Bc + col]);
        }

        // 在线 softmax：更新本行新的最大值
        float oldMax = Max[warp];
        float newMax = fmaxf(oldMax, rowMax);
        float expScale = __expf(oldMax - newMax);

        // 2) 把 score 转成 exp(score - newMax)
        //    同时求当前 tile 的 sum exp
        float rowSumExp = 0.0f;
        if (lane < Bc) { // Bc <= 32
            rowSumExp = __expf(s_S[warp * Bc + lane] - newMax);
            s_S[warp * Bc + lane] = rowSumExp;   // 原地覆盖为概率分子
        }

        // 注意：这里必须 warp 内求和
        rowSumExp = warp_reduce_sum(rowSumExp);
        float newSumExp = expScale * SumExp[warp] + rowSumExp;
        __syncwarp();

        // 3) 更新输出分子：
        //    s_O_new = exp(oldMax - newMax) * s_O_old + P_tile * V_tile
        for (int col = lane; col < d; col += 32) {
            float newO = expScale * s_O[warp * d + col];
#pragma unroll
            for (int i = 0; i < Bc; ++i) {
                newO += s_S[warp * Bc + i] * s_V[i * d + col];
            }
            s_O[warp * d + col] = newO;
        }

        // lane0 更新本行在线 softmax 状态
        if (lane == 0) {
            Max[warp] = newMax;
            SumExp[warp] = newSumExp;
        }

        // 下一轮会覆盖 s_K/s_V，先确保所有 warp 都用完当前 tile
        __syncthreads();
    }

    // 最后做归一化：O = s_O / SumExp
    for (int col = lane; col < d; col += 32) {
        O[offset + r * d + (warp * d + col)] = s_O[warp * d + col] / SumExp[warp];
    }
}

void launch_flash_attention_v2(float* Q, float* K, float* V, float* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;  // 一个 block 处理 16 行 Q
    constexpr int Bc = 16;  // 每次处理 16 列 K/V
    int sram_size = (2 * Br * d + 2 * Bc * d + Br * Bc + 2 * Br) * sizeof(float);
    dim3 gridDim(N / Br, heads, batch);
    dim3 blockDim(Br * 32);  // 一个 warp 对应一行 Q，所以总线程数 = Br * 32
    flash_attention_v2_kernel<Br, Bc><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version2


namespace fa2_version3 {
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, delta);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, delta));
    }
    return val;
}

template <int Br, int Bc, int Bd>
__global__ void flash_attention_v2_kernel(float* Q, float* K, float* V, float* O, int N, int d, float scale) {
    extern __shared__ float s_O[];
    __shared__ float s_Q[Br * Bd], s_K[Bc * Bd], s_V[Bc * Bd];
    __shared__ float s_S[Br * Bc];
    __shared__ float Max[Br], SumExp[Br];
    
    // 一个 (batch, head) 对应一张 attention[N, d] 矩阵
    int offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;

    // 当前 block 处理的 Q 起始行
    int row_q = blockIdx.x * Br;

    // 线程组织：1 warp 负责 1 行 query
    int lane = threadIdx.x & 31;   // warp 内 lane id
    int warp = threadIdx.x >> 5;   // 第几个 warp，对应第几行 query
    int warp_num = blockDim.x >> 5;

    // 把输出分子累加器 s_O 初始化为 0
    for (int i = threadIdx.x; i < Br * d; i += blockDim.x) {
        s_O[i] = 0.0f;
    }

    // 每个 warp 对应一行，初始化在线 softmax 状态
    if (lane == 0) {
        Max[warp] = -INFINITY;
        SumExp[warp] = 0.0f;
    }
    __syncthreads();

    // 遍历 K/V 的 sequence tile
    for (int j = 0; j < N; j += Bc) {
        // 每个 lane 对应当前 tile 的一个 key 列
        float score = 0.0f;

        // 1) 沿 d 维按 Bd 分块，累计 score = Q_row · K_col
        for (int k = 0; k < d; k += Bd) {
            for (int col = lane; col < Bd; col += 32) {
                s_Q[warp * Bd + col] = Q[offset + (row_q + warp) * d + (k + col)];
            }
            for (int row = warp; row < Bc; row += warp_num) {
                for (int col = lane; col < Bd; col += 32) {
                    s_K[row * Bd + col] = K[offset + (j + row) * d + (k + col)];
                }
            }
            __syncthreads();

            // warp 内：lane 负责当前 tile 的一个 key 列
#pragma unroll
            for (int i = 0; i < Bd; ++i) {
                score += s_Q[warp * Bd + i] * s_K[lane * Bd + i]; // 确保 Bc = 32
            }
            __syncthreads();
        }
        score *= scale;

        // 2) 当前 row 的 online softmax 更新
        float rowMax = warp_reduce_max(score);
        float oldMax = Max[warp];
        float newMax = fmaxf(oldMax, rowMax);
        float expScale = __expf(oldMax - newMax);

        float p = __expf(score - newMax);
        s_S[warp * Bc + lane] = p;
        float rowSumExp = warp_reduce_sum(p);
        float newSumExp = expScale * SumExp[warp] + rowSumExp;
        __syncwarp();

        // 3) 沿 d 维按 Bd 分块，更新 s_O 分子
        //    s_O_new = exp(oldMax - newMax) * s_O_old + P_tile * V_tile
        for (int k = 0; k < d; k += Bd) {
            for (int row = warp; row < Bc; row += warp_num) {
                for (int col = lane; col < Bd; col += 32) {
                    s_V[row * Bd + col] = V[offset + (j + row) * d + (k + col)];
                }
            }
            __syncthreads();

            // 每个 lane 负责当前输出 tile 中一列，确保 Bd = 32
            float newO = expScale * s_O[warp * d + (k + lane)];
#pragma unroll
            for (int i = 0; i < Bc; ++i) {
                newO += s_S[warp * Bc + i] * s_V[i * Bd + lane];
            }
            s_O[warp * d + (k + lane)] = newO;
            __syncthreads();
        }

        // lane0 更新本行在线 softmax 状态
        if (lane == 0) {
            Max[warp] = newMax;
            SumExp[warp] = newSumExp;
        }

        // 下一轮会覆盖 s_K/s_V，先确保所有 warp 都用完当前 tile
        __syncthreads();
    }

    // 4) 最后统一归一化：O = s_O / SumExp
    for (int col = lane; col < d; col += 32) {
        O[offset + row_q * d + (warp * d + col)] = s_O[warp * d + col] / SumExp[warp];
    }
}

void launch_flash_attention_v2(float* Q, float* K, float* V, float* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;  // 1 个 block 处理 16 行 Q，总共 16 个 warp
    constexpr int Bc = 32;  // 1 个 warp 的 32 个 lane 对应 32 个 key
    constexpr int Bd = 32;  // d 维 tile
    int sram_size = Br * d  * sizeof(float);
    dim3 gridDim(N / Br, heads, batch);
    dim3 blockDim(Br * 32);  // 16 warps，一个 warp 对应一行 Q，所以总线程数 = Br * 32
    flash_attention_v2_kernel<Br, Bc, Bd><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version3


namespace fa2_version4 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

template <int GroupSize = 16>
__device__ __forceinline__ MD warp_reduce_md(MD val) {
    float new_m;
#pragma unroll
    for (int delta = (GroupSize >> 1); delta > 0; delta >>= 1) {
        new_m = fmaxf(val.m, __shfl_xor_sync(0xffffffff, val.m, delta, GroupSize));
        val.d = val.d * __expf(val.m - new_m) + 
                __shfl_xor_sync(0xffffffff, val.d, delta, GroupSize) * __expf(__shfl_xor_sync(0xffffffff, val.m, delta, GroupSize) - new_m);
        val.m = new_m;
    }
    return val;
}

#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define MMA_M16N8K16_F16F16F16F16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define LDST_32_BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST_128_BITS(value) (reinterpret_cast<float4*>(&(value))[0])

/*
每个 block 包含 4 个 warp，每个 warp 单独处理 [Br, d] 的 Q 矩阵分片，4 个 warp 共用 [Br, d] 的 K、V 分片
gridDim(N / (4 * Br), head_num, batch_size)
blockDim(128)
Q / O: [batch_size, head_num, N, d]
K / V: [batch_size, head_num, N, d]
*/
template <int Br, int Bc, int Bd>
__global__ void flash_attention_v2_kernel(half* Q, half* K, half* V, half* O, int N, int d, float scale) {
    uint32_t warp = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31;

    uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d;
    uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;

    extern __shared__ half sram[];
    half* s_Q = sram;
    half* s_K = s_Q + 4 * Br * d;
    half* s_V = s_K + Bc * d;
    half* s_S = s_V + Bc * d;
    half* s_O = s_S + 4 * Br * Bc;
    MD* row_ml_old = reinterpret_cast<MD*>(s_O + 4 * Br * Bd);
    MD* row_ml_new = row_ml_old + 4 * Br;

    if (lane < Br) {
        row_ml_old[warp * Br + lane] = { -INFINITY, 0.0f };
    }

    // load [4 * Br, d] 的 Q 矩阵分片到 s_Q，每个 warp load [Br, d]，每次 load 8 个 half
    for (int i = (lane << 3); i < Br * d; i += (32 << 3)) {
        LDST_128_BITS(s_Q[warp * Br * d + i]) = LDST_128_BITS(Q[qo_offset + warp * Br * d + i]);
    }
    __syncwarp();

    // warp 矩阵乘法的尺寸为 16 × 16 × 16
    // 调用两次 mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 指令
    // 所以 3 个矩阵都需要 4 个 32-bit 寄存器，因为 ldmatrix.x4 加载 4 个 8 × 8 子矩阵，每个线程会得到 4 个 32-bit 寄存器结果
    uint32_t RA[4];  // 从 s_Q 中加载出来、供 mma 使用的 A 矩阵片段
    uint32_t RB[4];  // 从 s_K 中加载出来、供 mma 使用的 B 矩阵片段

    // 对 K/V 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int i = 0; i < N; i += Bc) {
        // 初始化矩阵 C 的寄存器
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // load [Bc, d] 的 K/V 矩阵分片到 s_K/s_V，整个 block 一起 load [Bc, d]，每次 load 8 个 half
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            LDST_128_BITS(s_K[j]) = LDST_128_BITS(K[kv_offset + i * d + j]);
            LDST_128_BITS(s_V[j]) = LDST_128_BITS(V[kv_offset + i * d + j]);
        }
        __syncthreads();

        // 计算 S = QK^T 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 从 s_Q[4 * Br, d] load 16 × 16 矩阵分片到 RA，每个 warp 负责 [Br, Bd] 分片
            // warp 内每个线程都需要传入一个地址: ldmatrix 是 warp 协同指令，整个 warp 一起合作装载一个 mma 所需的 tile
            // warp 内线程 0-7   加载第 1 个左上角 8 × 8 矩阵，坐标范围是 [0-7,  0]
            // warp 内线程 8-15  加载第 2 个左下角 8 × 8 矩阵，坐标范围是 [8-15, 0]
            // warp 内线程 16-23 加载第 3 个右上角 8 × 8 矩阵，坐标范围是 [0-7,  8]
            // warp 内线程 24-31 加载第 4 个右下角 8 × 8 矩阵，坐标范围是 [8-15, 8]
            // 因此线程 lane 对应的坐标是 (lane % 16, lane / 16)
            // 偏移量计算: (lane % 16) * d + (lane / 16) * 8
            uint32_t addr = (warp * Br * d) + k + (lane & 15) * d + (lane >> 4) * 8;
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + addr);

            // 从行主序的 s_K[Bc, d]（s_K^T 的列主序）load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            // warp 内线程 0-7   加载第 1 个左上角 8 × 8 矩阵，坐标范围是 [0-7,  0]
            // warp 内线程 8-15  加载第 2 个右上角 8 × 8 矩阵，坐标范围是 [0-7,  8]
            // warp 内线程 16-23 加载第 3 个左下角 8 × 8 矩阵，坐标范围是 [8-15, 0]
            // warp 内线程 24-31 加载第 4 个右下角 8 × 8 矩阵，坐标范围是 [8-15, 8]
            // 因此线程 lane 对应的坐标是 ((lane / 16) * 8 + (lane % 8), (lane / 8) % 2)
            // 偏移量计算: ((lane / 16) * 8 + (lane % 8)) * d + ((lane / 8) % 2) * 8
            addr = k + ((lane >> 4) * 8 + (lane & 7)) * d + ((lane >> 3) & 1) * 8;
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + addr);

            // 我们想要计算 Q tile (16 × 16) 和 K^T tile (16 × 16) 相乘的结果
            // 但是把 K^T tile (16 × 16) 拆成了 2 个 16 × 8 左右子矩阵
            // 因为寄存器是列主序存储的，因此左对应 (RB[0], RB[1])，右对应了 (RC[2], RC[3])
            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            __syncwarp();
        }

        // 将矩阵 C 的寄存器变量写入 s_S[4 * Br, Bc]，每个 warp 仅负责 [Br, Bc] 分片，sm_90 之前不支持 stmatrix 指令 (RTX 4090 不支持)
        // 每个 8 × 8 子矩阵按列主序填充 RC[0/1/2/3]，参照 mma 指令规定的矩阵 C 的元素排布，每次写入 32-bit
        // 矩阵 C 每个 8 × 8 子矩阵内部按照行主序填充，在每个子矩阵中第 0 行 T[0,1,2,3]，第 1 行 T[4,5,6,7] ... 以此类推
        // 每个线程 RC[0] = (a0, a1)，RC[1] = (a2, a3)，RC[2] = (a4, a5)，RC[3] = (a6, a7)
        // 每个线程在每个子矩阵中对应 2 个 half 元素，比如子矩阵 B0 中第 0 行 T0(a0, a1), T1(a0, a1), T2(a0, a1), T3(a0, a1)
        // 因此，线程 lane 在每个子矩阵内坐标是 (lane / 4, lane % 4)
        // 偏移量计算: (lane / 4) * Bc + (lane % 4) * 2
        LDST_32_BITS(s_S[(warp * Br * Bc) + (lane >> 2) * Bc       + (lane & 3) * 2]) = LDST_32_BITS(RC[0]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + ((lane >> 2) + 8) * Bc + (lane & 3) * 2]) = LDST_32_BITS(RC[1]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (lane >> 2) * Bc       + (lane & 3) * 2 + 8]) = LDST_32_BITS(RC[2]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + ((lane >> 2) + 8) * Bc + (lane & 3) * 2 + 8]) = LDST_32_BITS(RC[3]);
        __syncwarp();

        // 对 s_S 求 softmax，每个 warp 单独计算 [Br, Bc] = [16, 16] 矩阵的 softmax，根据 online softmax 先计算 m 和 l
        // 1 个 warp 每次单独处理 2 行，每行 16 个元素，在 warp 内的 16 个线程内部做规约，总共需要处理 Br / 2 = 8 次
#pragma unroll
        for (int j = 0; j < (Br >> 1); ++j) {
            // 读取 2 行数据到 warp
            MD temp_ml = { __half2float(s_S[warp * Br * Bc + j * 32 + lane]) * scale, 1.0f };

            // 每行数据由 16 个线程组成的 group 持有，内部 reduce
            temp_ml = warp_reduce_md<16>(temp_ml);

            // 当前线程处理的行索引
            uint32_t row = warp * Br + j * 2 + (lane >> 4);
            if ((lane & 15) == 0) { // lane = 0 or 16
                row_ml_new[row] = MD_OP()(row_ml_old[row], temp_ml);
            }
            __syncwarp();

            s_S[row * Bc + (lane & 15)] = __float2half(
                __expf(__half2float(s_S[row * Bc + (lane & 15)]) * scale - row_ml_new[row].m)
            );
        }
        
        // 从 s_S[4 * Br, Bc] load 16 × 16 矩阵分片到 RA，每个 warp 仅负责 [Br, Bc] 分片
        // warp 内线程 0-7   加载第 1 个左上角 8 × 8 矩阵，坐标范围是 [0-7,  0]
        // warp 内线程 8-15  加载第 2 个左下角 8 × 8 矩阵，坐标范围是 [8-15, 0]
        // warp 内线程 16-23 加载第 3 个右上角 8 × 8 矩阵，坐标范围是 [0-7,  8]
        // warp 内线程 24-31 加载第 4 个右下角 8 × 8 矩阵，坐标范围是 [8-15, 8]
        // 因此线程 lane 对应的坐标是 (lane % 16, lane / 16)
        // 偏移量计算: (lane % 16) * Bc + (lane / 16) * 8
        uint32_t addr = (warp * Br * Bc) + (lane & 15) * Bc + (lane >> 4) * 8;
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + addr);

        // 计算 O = PV 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 初始化矩阵 C 的寄存器
            RC[0] = RC[1] = RC[2] = RC[3] = 0;

            // 从 s_V[Bc, d] load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            // warp 内线程 0-7   加载第 1 个左上角 8 × 8 矩阵，坐标范围是 [0-7,  0]
            // warp 内线程 8-15  加载第 2 个左下角 8 × 8 矩阵，坐标范围是 [8-15, 0]
            // warp 内线程 16-23 加载第 3 个右上角 8 × 8 矩阵，坐标范围是 [0-7,  8]
            // warp 内线程 24-31 加载第 4 个右下角 8 × 8 矩阵，坐标范围是 [8-15, 8]
            // 因此线程 lane 对应的坐标是 (lane % 16, lane / 16)
            // 偏移量计算: (lane % 16) * d + (lane / 16) * 8
            // 由于右乘矩阵 RB 中每个子矩阵是列主序的，因此实际加载的子矩阵数据排布是 V^T，需要转置 trans
            addr = k + (lane & 15) * d + (lane >> 4) * 8;
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 将矩阵 C 的寄存器变量写入 s_O[4 * Br, Bd]，每个 warp 仅负责 [Br, Bd] 分片，sm_90 之前不支持 stmatrix 指令 (RTX 4090 不支持)
            // 每个 8 × 8 子矩阵按列主序填充 RC[0/1/2/3]，参照 mma 指令规定的矩阵 C 的元素排布，每次写入 32-bit
            // 矩阵 C 每个 8 × 8 子矩阵内部按照行主序填充，在每个子矩阵中第 0 行 T[0,1,2,3]，第 1 行 T[4,5,6,7] ... 以此类推
            // 每个线程 RC[0] = (a0, a1)，RC[1] = (a2, a3)，RC[2] = (a4, a5)，RC[3] = (a6, a7)
            // 每个线程在每个子矩阵中对应 2 个 half 元素，比如子矩阵 B0 中第 0 行 T0(a0, a1), T1(a0, a1), T2(a0, a1), T3(a0, a1)
            // 因此，线程 lane 在每个子矩阵内坐标是 (lane / 4, lane % 4)
            // 偏移量计算: (lane / 4) * d + (lane % 4) * 2
            LDST_32_BITS(s_O[(warp * Br * Bd) + (lane >> 2) * Bd       + (lane & 3) * 2]) = LDST_32_BITS(RC[0]);
            LDST_32_BITS(s_O[(warp * Br * Bd) + ((lane >> 2) + 8) * Bd + (lane & 3) * 2]) = LDST_32_BITS(RC[1]);
            LDST_32_BITS(s_O[(warp * Br * Bd) + (lane >> 2) * Bd       + (lane & 3) * 2 + 8]) = LDST_32_BITS(RC[2]);
            LDST_32_BITS(s_O[(warp * Br * Bd) + ((lane >> 2) + 8) * Bd + (lane & 3) * 2 + 8]) = LDST_32_BITS(RC[3]);
            __syncwarp();

            // 更新 O 中 s_O[4 * Br, Bd] 分块，每个 warp 单独负责 [Br, Bd] = [16, 16] 分片
            // 1 个 warp 每次单独处理 2 行，每行 16 个元素，在 warp 内的 16 个线程内部做规约，总共需要处理 Br / 2 = 8 次
#pragma unroll
            for (int j = 0; j < (Br >> 1); ++j) {
                // 当前线程在 s_O[4 * Br, Bd] 的行索引
                uint32_t row = warp * Br + j * 2 + (lane >> 4);
                // 当前线程在 s_O[4 * Br, Bd] 的索引
                uint32_t s_o_idx = row * Bd + (lane & 15);
                // 当前线程在全局 O[batch, heads, N, d] 的索引
                uint32_t o_idx = qo_offset + k + row * d + (lane & 15);

                O[o_idx] = __float2half((
                    row_ml_old[row].d * __expf(row_ml_old[row].m - row_ml_new[row].m) * __half2float(O[o_idx]) + 
                    __half2float(s_O[s_o_idx])) / row_ml_new[row].d
                );
            }
        }

        // 更新 row_ml_old
        if (lane < Br) {
            row_ml_old[warp * Br + lane] = row_ml_new[warp * Br + lane];
        }
        __syncthreads();
    }
}

void launch_flash_attention_v2(half* Q, half* K, half* V, half* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int Bd = 16;
    int sram_size = (4 * Br * d + 2 * Bc * d + 4 * Br * Bc + 4 * Br * Bd) * sizeof(half) + (8 * Br) * sizeof(MD);
    dim3 gridDim(N / (4 * Br), heads, batch);
    dim3 blockDim(128);
    flash_attention_v2_kernel<Br, Bc, Bd><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version4


namespace fa2_version5 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

template <int GroupSize = 16>
__device__ __forceinline__ MD warp_reduce_md(MD val) {
    float new_m;
#pragma unroll
    for (int delta = (GroupSize >> 1); delta > 0; delta >>= 1) {
        new_m = fmaxf(val.m, __shfl_xor_sync(0xffffffff, val.m, delta, GroupSize));
        val.d = val.d * __expf(val.m - new_m) + 
                __shfl_xor_sync(0xffffffff, val.d, delta, GroupSize) * __expf(__shfl_xor_sync(0xffffffff, val.m, delta, GroupSize) - new_m);
        val.m = new_m;
    }
    return val;
}

#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define MMA_M16N8K16_F16F16F16F16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define LDST_32_BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST_128_BITS(value) (reinterpret_cast<float4*>(&(value))[0])

/**
 * \tparam S: SShift, right shift the addr for swizzling
 * \tparam B: BShift, bits to be swizzled
 * \tparam M: MBase, bits keep the same
 */
template <uint32_t B, uint32_t M, uint32_t S>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    // addr = (1 << (M + S)) * row + (1 << M) * col
    // swizzle(addr) = (1 << (M + S)) * row + (1 << M) * ((row % (1 << B)) ^ col)
    constexpr uint32_t Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}

/*
在 fa2_version4 基础上增加 swizzle 策略规避 bank conflict
gridDim(N / (4 * Br), head_num, batch_size)
blockDim(128)
Q / O: [batch_size, head_num, N, d]
K / V: [batch_size, head_num, N, d]
*/
template <int Br, int Bc, int Bd>
__global__ void flash_attention_v2_kernel(half* Q, half* K, half* V, half* O, int N, int d, float scale) {
    uint32_t warp = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31;

    uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d;
    uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;

    extern __shared__ half sram[];
    half* s_Q = sram;
    half* s_K = s_Q + 4 * Br * d;
    half* s_V = s_K + Bc * d;
    half* s_S = s_V + Bc * d;
    half* s_O = s_S + 4 * Br * Bc;
    MD* row_ml_old = reinterpret_cast<MD*>(s_O + 4 * Br * Bd);
    MD* row_ml_new = row_ml_old + 4 * Br;

    if (lane < Br) {
        row_ml_old[warp * Br + lane] = { -INFINITY, 0.0f };
    }

    // load [4 * Br, d] 的 Q 矩阵分片到 s_Q，每个 warp load [Br, d]，每次 load 8 个 half
    for (int i = (lane << 3); i < Br * d; i += (32 << 3)) {
        LDST_128_BITS(s_Q[warp * Br * d + (swizzle<3, 3, 4>(i))]) = LDST_128_BITS(Q[qo_offset + warp * Br * d + i]);
    }
    __syncwarp();

    uint32_t RA[4];
    uint32_t RB[4];

    // 对 K/V 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int i = 0; i < N; i += Bc) {
        // 初始化矩阵 C 的寄存器
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // load [Bc, d] 的 K/V 矩阵分片到 s_K/s_V，整个 block 一起 load [Bc, d]，每次 load 8 个 half
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            LDST_128_BITS(s_K[(swizzle<3, 3, 4>(j))]) = LDST_128_BITS(K[kv_offset + i * d + j]);
            LDST_128_BITS(s_V[(swizzle<3, 3, 4>(j))]) = LDST_128_BITS(V[kv_offset + i * d + j]);
        }
        __syncthreads();

        // 计算 S = QK^T 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 从 s_Q[4 * Br, d] load 16 × 16 矩阵分片到 RA，每个 warp 负责 [Br, Bd] 分片
            uint32_t addr = (warp * Br * d) + swizzle<3, 3, 4>(k + (lane & 15) * d + (lane >> 4) * 8);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + addr);

            // 从行主序的 s_K[Bc, d]（s_K^T 的列主序）load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            addr = swizzle<3, 3, 4>(k + ((lane >> 4) * 8 + (lane & 7)) * d + ((lane >> 3) & 1) * 8);
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            __syncwarp();
        }

        // 将矩阵 C 的寄存器变量写入 s_S[4 * Br, Bc]，每个 warp 仅负责 [Br, Bc] 分片，sm_90 之前不支持 stmatrix 指令 (RTX 4090 不支持)
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>((lane >> 2) * Bc       + (lane & 3) * 2))]) = LDST_32_BITS(RC[0]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>(((lane >> 2) + 8) * Bc + (lane & 3) * 2))]) = LDST_32_BITS(RC[1]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>((lane >> 2) * Bc       + (lane & 3) * 2 + 8))]) = LDST_32_BITS(RC[2]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>(((lane >> 2) + 8) * Bc + (lane & 3) * 2 + 8))]) = LDST_32_BITS(RC[3]);
        __syncwarp();

        // 对 s_S 求 softmax，每个 warp 单独计算 [Br, Bc] = [16, 16] 矩阵的 softmax，根据 online softmax 先计算 m 和 l
        // 1 个 warp 每次单独处理 2 行，每行 16 个元素，在 warp 内的 16 个线程内部做规约，总共需要处理 Br / 2 = 8 次
#pragma unroll
        for (int j = 0; j < (Br >> 1); ++j) {
            // 读取 2 行数据到 warp
            MD temp_ml = { __half2float(s_S[warp * Br * Bc + j * 32 + lane]) * scale, 1.0f };

            // 每行数据由 16 个线程组成的 group 持有，行内 reduce
            temp_ml = warp_reduce_md<16>(temp_ml);

            // 当前线程处理的行索引
            uint32_t row = warp * Br + j * 2 + (lane >> 4);
            if ((lane & 15) == 0) { // lane = 0 or 16
                row_ml_new[row] = MD_OP()(row_ml_old[row], temp_ml);
            }
            __syncwarp();

            // 行内逐元素更新
            s_S[row * Bc + (lane & 15)] = __float2half(
                __expf(__half2float(s_S[row * Bc + (lane & 15)]) * scale - row_ml_new[row].m)
            );
        }
        
        // 从 s_S[4 * Br, Bc] load 16 × 16 矩阵分片到 RA，每个 warp 仅负责 [Br, Bc] 分片
        uint32_t addr = (warp * Br * Bc) + swizzle<1, 3, 3>((lane & 15) * Bc + (lane >> 4) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + addr);

        // 计算 O = PV 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 初始化矩阵 C 的寄存器
            RC[0] = RC[1] = RC[2] = RC[3] = 0;

            // 从 s_V[Bc, d] load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            addr = swizzle<3, 3, 4>(k + (lane & 15) * d + (lane >> 4) * 8);
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 将矩阵 C 的寄存器变量写入 s_O[4 * Br, Bd]，每个 warp 仅负责 [Br, Bd] 分片，sm_90 之前不支持 stmatrix 指令 (RTX 4090 不支持)
            LDST_32_BITS(s_O[(warp * Br * Bd) + (swizzle<1, 3, 3>((lane >> 2) * Bd       + (lane & 3) * 2))]) = LDST_32_BITS(RC[0]);
            LDST_32_BITS(s_O[(warp * Br * Bd) + (swizzle<1, 3, 3>(((lane >> 2) + 8) * Bd + (lane & 3) * 2))]) = LDST_32_BITS(RC[1]);
            LDST_32_BITS(s_O[(warp * Br * Bd) + (swizzle<1, 3, 3>((lane >> 2) * Bd       + (lane & 3) * 2 + 8))]) = LDST_32_BITS(RC[2]);
            LDST_32_BITS(s_O[(warp * Br * Bd) + (swizzle<1, 3, 3>(((lane >> 2) + 8) * Bd + (lane & 3) * 2 + 8))]) = LDST_32_BITS(RC[3]);
            __syncwarp();

            // 更新 O 中 s_O[4 * Br, Bd] 分块，每个 warp 单独负责 [Br, Bd] = [16, 16] 分片
            // 1 个 warp 每次单独处理 2 行，每行 16 个元素，在 warp 内的 16 个线程内部做规约，总共需要处理 Br / 2 = 8 次
#pragma unroll
            for (int j = 0; j < (Br >> 1); ++j) {
                // 当前线程在 s_O[4 * Br, Bd] 的行索引
                uint32_t row = warp * Br + j * 2 + (lane >> 4);
                // 当前线程在 s_O[4 * Br, Bd] 的索引
                uint32_t s_o_idx = (warp * Br * Bd) + swizzle<1, 3, 3>((j * 2 + (lane >> 4)) * Bd + (lane & 15));
                // 当前线程在全局 O[batch, heads, N, d] 的索引
                uint32_t o_idx = qo_offset + k + row * d + (lane & 15);

                O[o_idx] = __float2half((
                    row_ml_old[row].d * __expf(row_ml_old[row].m - row_ml_new[row].m) * __half2float(O[o_idx]) + 
                    __half2float(s_O[s_o_idx])) / row_ml_new[row].d
                );
            }
        }

        // 更新 row_ml_old
        if (lane < Br) {
            row_ml_old[warp * Br + lane] = row_ml_new[warp * Br + lane];
        }
        __syncthreads();
    }
}

void launch_flash_attention_v2(half* Q, half* K, half* V, half* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int Bd = 16;
    int sram_size = (4 * Br * d + 2 * Bc * d + 4 * Br * Bc + 4 * Br * Bd) * sizeof(half) + (8 * Br) * sizeof(MD);
    dim3 gridDim(N / (4 * Br), heads, batch);
    dim3 blockDim(128);
    flash_attention_v2_kernel<Br, Bc, Bd><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version5


namespace fa2_version6 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

template <int GroupSize = 16>
__device__ __forceinline__ MD warp_reduce_md(MD val) {
    float new_m;
#pragma unroll
    for (int delta = (GroupSize >> 1); delta > 0; delta >>= 1) {
        new_m = fmaxf(val.m, __shfl_xor_sync(0xffffffff, val.m, delta, GroupSize));
        val.d = val.d * __expf(val.m - new_m) + 
                __shfl_xor_sync(0xffffffff, val.d, delta, GroupSize) * __expf(__shfl_xor_sync(0xffffffff, val.m, delta, GroupSize) - new_m);
        val.m = new_m;
    }
    return val;
}

#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define MMA_M16N8K16_F16F16F16F16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define LDST_32_BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST_128_BITS(value) (reinterpret_cast<float4*>(&(value))[0])

/*
在 fa2_version4 基础上修改了最终更新 O 的逻辑
gridDim(N / (4 * Br), head_num, batch_size)
blockDim(128)
Q / O: [batch_size, head_num, N, d]
K / V: [batch_size, head_num, N, d]
*/
template <int Br, int Bc, int Bd>
__global__ void flash_attention_v2_kernel(half* Q, half* K, half* V, half* O, int N, int d, float scale) {
    uint32_t warp = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31;

    uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d;
    uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;

    extern __shared__ half sram[];
    half* s_Q = sram;
    half* s_K = s_Q + 4 * Br * d;
    half* s_V = s_K + Bc * d;
    half* s_S = s_V + Bc * d;
    half* s_O = s_S + 4 * Br * Bc; // 由 [4 * Br, Bd] 改成 [4 * Br, d]
    MD* row_ml_old = reinterpret_cast<MD*>(s_O + 4 * Br * d);
    MD* row_ml_new = row_ml_old + 4 * Br;

    // 进入主循环前，把 s_O 清零
    for (int i = threadIdx.x; i < 4 * Br * d; i += blockDim.x) {
        s_O[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (lane < Br) {
        row_ml_old[warp * Br + lane] = { -INFINITY, 0.0f };
    }

    // load [4 * Br, d] 的 Q 矩阵分片到 s_Q，每个 warp load [Br, d]，每次 load 8 个 half
    for (int i = (lane << 3); i < Br * d; i += (32 << 3)) {
        LDST_128_BITS(s_Q[warp * Br * d + i]) = LDST_128_BITS(Q[qo_offset + warp * Br * d + i]);
    }
    __syncwarp();

    uint32_t RA[4];
    uint32_t RB[4];

    // 对 K/V 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int i = 0; i < N; i += Bc) {
        // 初始化矩阵 C 的寄存器
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // load [Bc, d] 的 K/V 矩阵分片到 s_K/s_V，整个 block 一起 load [Bc, d]，每次 load 8 个 half
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            LDST_128_BITS(s_K[j]) = LDST_128_BITS(K[kv_offset + i * d + j]);
            LDST_128_BITS(s_V[j]) = LDST_128_BITS(V[kv_offset + i * d + j]);
        }
        __syncthreads();

        // 计算 S = QK^T 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 从 s_Q[4 * Br, d] load 16 × 16 矩阵分片到 RA，每个 warp 负责 [Br, Bd] 分片
            uint32_t addr = (warp * Br * d) + k + (lane & 15) * d + (lane >> 4) * 8;
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + addr);

            // 从行主序的 s_K[Bc, d]（s_K^T 的列主序）load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            addr = k + ((lane >> 4) * 8 + (lane & 7)) * d + ((lane >> 3) & 1) * 8;
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            __syncwarp();
        }

        // 将矩阵 C 的寄存器变量写入 s_S[4 * Br, Bc]，每个 warp 仅负责 [Br, Bc] 分片，sm_90 之前不支持 stmatrix 指令 (RTX 4090 不支持)
        LDST_32_BITS(s_S[(warp * Br * Bc) + (lane >> 2) * Bc       + (lane & 3) * 2]) = LDST_32_BITS(RC[0]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + ((lane >> 2) + 8) * Bc + (lane & 3) * 2]) = LDST_32_BITS(RC[1]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (lane >> 2) * Bc       + (lane & 3) * 2 + 8]) = LDST_32_BITS(RC[2]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + ((lane >> 2) + 8) * Bc + (lane & 3) * 2 + 8]) = LDST_32_BITS(RC[3]);
        __syncwarp();

        // 对 s_S 求 softmax，每个 warp 单独计算 [Br, Bc] = [16, 16] 矩阵的 softmax，根据 online softmax 先计算 m 和 l
        // 1 个 warp 每次单独处理 2 行，每行 16 个元素，在 warp 内的 16 个线程内部做规约，总共需要处理 Br / 2 = 8 次
#pragma unroll
        for (int j = 0; j < (Br >> 1); ++j) {
            // 读取 2 行数据到 warp
            MD temp_ml = { __half2float(s_S[warp * Br * Bc + j * 32 + lane]) * scale, 1.0f };

            // 每行数据由 16 个线程组成的 group 持有，内部 reduce
            temp_ml = warp_reduce_md<16>(temp_ml);

            // 当前线程处理的行索引
            uint32_t row = warp * Br + j * 2 + (lane >> 4);
            if ((lane & 15) == 0) { // lane = 0 or 16
                row_ml_new[row] = MD_OP()(row_ml_old[row], temp_ml);
            }
            __syncwarp();

            s_S[row * Bc + (lane & 15)] = __float2half(
                __expf(__half2float(s_S[row * Bc + (lane & 15)]) * scale - row_ml_new[row].m)
            );
        }
        
        // 从 s_S[4 * Br, Bc] load 16 × 16 矩阵分片到 RA，每个 warp 仅负责 [Br, Bc] 分片
        uint32_t addr = (warp * Br * Bc) + (lane & 15) * Bc + (lane >> 4) * 8;
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + addr);

        // 计算 O = PV 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 初始化矩阵 C 的寄存器
            RC[0] = RC[1] = RC[2] = RC[3] = 0;

            // 从 s_V[Bc, d] load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            addr = k + (lane & 15) * d + (lane >> 4) * 8;
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 当前线程对应的两组行：row0 / row1
            uint32_t row0 = warp * Br + (lane >> 2);
            uint32_t row1 = row0 + 8;

            // 当前线程对应的列起点（每次处理 half2）
            uint32_t col0 = k + (lane & 3) * 2;

            // s_O 是 [4 * Br, d] 的完整分子累加器
            uint32_t idx0 = row0 * d + col0;
            uint32_t idx1 = row1 * d + col0;
            uint32_t idx2 = row0 * d + col0 + 8;
            uint32_t idx3 = row1 * d + col0 + 8;

            // 4 个 half2 结果
            half2 cur0_h2, cur1_h2, cur2_h2, cur3_h2;
            LDST_32_BITS(cur0_h2) = RC[0];
            LDST_32_BITS(cur1_h2) = RC[1];
            LDST_32_BITS(cur2_h2) = RC[2];
            LDST_32_BITS(cur3_h2) = RC[3];

            float alpha0f = __expf(row_ml_old[row0].m - row_ml_new[row0].m);
            float alpha1f = __expf(row_ml_old[row1].m - row_ml_new[row1].m);

            half2 old0_h2 = LDST_32_BITS(s_O[idx0]);
            half2 old1_h2 = LDST_32_BITS(s_O[idx1]);
            half2 old2_h2 = LDST_32_BITS(s_O[idx2]);
            half2 old3_h2 = LDST_32_BITS(s_O[idx3]);

            float2 old0_f2 = __half22float2(old0_h2);
            float2 old1_f2 = __half22float2(old1_h2);
            float2 old2_f2 = __half22float2(old2_h2);
            float2 old3_f2 = __half22float2(old3_h2);

            float2 cur0_f2 = __half22float2(cur0_h2);
            float2 cur1_f2 = __half22float2(cur1_h2);
            float2 cur2_f2 = __half22float2(cur2_h2);
            float2 cur3_f2 = __half22float2(cur3_h2);

            old0_f2.x = fmaf(alpha0f, old0_f2.x, cur0_f2.x);
            old0_f2.y = fmaf(alpha0f, old0_f2.y, cur0_f2.y);

            old1_f2.x = fmaf(alpha1f, old1_f2.x, cur1_f2.x);
            old1_f2.y = fmaf(alpha1f, old1_f2.y, cur1_f2.y);

            old2_f2.x = fmaf(alpha0f, old2_f2.x, cur2_f2.x);
            old2_f2.y = fmaf(alpha0f, old2_f2.y, cur2_f2.y);

            old3_f2.x = fmaf(alpha1f, old3_f2.x, cur3_f2.x);
            old3_f2.y = fmaf(alpha1f, old3_f2.y, cur3_f2.y);

            LDST_32_BITS(s_O[idx0]) = __float22half2_rn(old0_f2);
            LDST_32_BITS(s_O[idx1]) = __float22half2_rn(old1_f2);
            LDST_32_BITS(s_O[idx2]) = __float22half2_rn(old2_f2);
            LDST_32_BITS(s_O[idx3]) = __float22half2_rn(old3_f2);

            __syncwarp();
        }

        // 更新 row_ml_old
        if (lane < Br) {
            row_ml_old[warp * Br + lane] = row_ml_new[warp * Br + lane];
        }
        __syncthreads();
    }

    for (int i = (threadIdx.x << 1); i < 4 * Br * d; i += (blockDim.x << 1)) {
        float2 val_f2 = __half22float2(LDST_32_BITS(s_O[i]));
        val_f2.x /= row_ml_new[i / d].d;
        val_f2.y /= row_ml_new[i / d].d;
        LDST_32_BITS(O[qo_offset + i]) = __float22half2_rn(val_f2);
    }
}

void launch_flash_attention_v2(half* Q, half* K, half* V, half* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int Bd = 16;
    int sram_size = (8 * Br * d + 2 * Bc * d + 4 * Br * Bc) * sizeof(half) + (8 * Br) * sizeof(MD);
    dim3 gridDim(N / (4 * Br), heads, batch);
    dim3 blockDim(128);
    flash_attention_v2_kernel<Br, Bc, Bd><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version6


namespace fa2_version7 {
struct __align__(8) MD {
    float m; // max val
    float d; // exp sum
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

template <int GroupSize = 16>
__device__ __forceinline__ MD warp_reduce_md(MD val) {
    float new_m;
#pragma unroll
    for (int delta = (GroupSize >> 1); delta > 0; delta >>= 1) {
        new_m = fmaxf(val.m, __shfl_xor_sync(0xffffffff, val.m, delta, GroupSize));
        val.d = val.d * __expf(val.m - new_m) + 
                __shfl_xor_sync(0xffffffff, val.d, delta, GroupSize) * __expf(__shfl_xor_sync(0xffffffff, val.m, delta, GroupSize) - new_m);
        val.m = new_m;
    }
    return val;
}

#define LDMATRIX_X4(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr) asm volatile("ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "l"(__cvta_generic_to_shared(addr)))
#define MMA_M16N8K16_F16F16F16F16(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n" : "=r"(RD0), "=r"(RD1) : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1))
#define LDST_32_BITS(value) (reinterpret_cast<half2*>(&(value))[0])
#define LDST_128_BITS(value) (reinterpret_cast<float4*>(&(value))[0])

/**
 * \tparam S: SShift, right shift the addr for swizzling
 * \tparam B: BShift, bits to be swizzled
 * \tparam M: MBase, bits keep the same
 */
template <uint32_t B, uint32_t M, uint32_t S>
__device__ __forceinline__ uint32_t swizzle(uint32_t addr) {
    // addr = (1 << (M + S)) * row + (1 << M) * col
    // swizzle(addr) = (1 << (M + S)) * row + (1 << M) * ((row % (1 << B)) ^ col)
    constexpr uint32_t Bmask = ((1 << B) - 1) << M;
    return ((addr >> S) & Bmask) ^ addr;
}

template <int Br, int Bc, int Bd>
__global__ void flash_attention_v2_kernel(half* Q, half* K, half* V, half* O, int N, int d, float scale) {
    uint32_t warp = threadIdx.x >> 5;
    uint32_t lane = threadIdx.x & 31;

    uint32_t qo_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d + blockIdx.x * 4 * Br * d;
    uint32_t kv_offset = (blockIdx.z * gridDim.y + blockIdx.y) * N * d;

    extern __shared__ half sram[];
    half* s_Q = sram;
    half* s_K = s_Q + 4 * Br * d;
    half* s_V = s_K + Bc * d;
    half* s_S = s_V + Bc * d;
    half* s_O = s_S + 4 * Br * Bc; // 由 [4 * Br, Bd] 改成 [4 * Br, d]
    MD* row_ml_old = reinterpret_cast<MD*>(s_O + 4 * Br * d);
    MD* row_ml_new = row_ml_old + 4 * Br;

    // 进入主循环前，把 s_O 清零
    for (int i = threadIdx.x; i < 4 * Br * d; i += blockDim.x) {
        s_O[i] = __float2half(0.0f);
    }
    __syncthreads();

    if (lane < Br) {
        row_ml_old[warp * Br + lane] = { -INFINITY, 0.0f };
    }

    // load [4 * Br, d] 的 Q 矩阵分片到 s_Q，每个 warp load [Br, d]，每次 load 8 个 half
    for (int i = (lane << 3); i < Br * d; i += (32 << 3)) {
        LDST_128_BITS(s_Q[warp * Br * d + (swizzle<3, 3, 4>(i))]) = LDST_128_BITS(Q[qo_offset + warp * Br * d + i]);
    }
    __syncwarp();

    uint32_t RA[4];
    uint32_t RB[4];

    // 对 K/V 在 N 维度分组，每组长度为 Bc，共分为 Tc = N / Bc 组
    for (int i = 0; i < N; i += Bc) {
        // 初始化矩阵 C 的寄存器
        uint32_t RC[4] = { 0, 0, 0, 0 };

        // load [Bc, d] 的 K/V 矩阵分片到 s_K/s_V，整个 block 一起 load [Bc, d]，每次 load 8 个 half
        for (int j = (threadIdx.x << 3); j < Bc * d; j += (blockDim.x << 3)) {
            LDST_128_BITS(s_K[(swizzle<3, 3, 4>(j))]) = LDST_128_BITS(K[kv_offset + i * d + j]);
            LDST_128_BITS(s_V[(swizzle<3, 3, 4>(j))]) = LDST_128_BITS(V[kv_offset + i * d + j]);
        }
        __syncthreads();

        // 计算 S = QK^T 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 从 s_Q[4 * Br, d] load 16 × 16 矩阵分片到 RA，每个 warp 负责 [Br, Bd] 分片
            uint32_t addr = (warp * Br * d) + swizzle<3, 3, 4>(k + (lane & 15) * d + (lane >> 4) * 8);
            LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_Q + addr);

            // 从行主序的 s_K[Bc, d]（s_K^T 的列主序）load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            addr = swizzle<3, 3, 4>(k + ((lane >> 4) * 8 + (lane & 7)) * d + ((lane >> 3) & 1) * 8);
            LDMATRIX_X4(RB[0], RB[1], RB[2], RB[3], s_K + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);
            __syncwarp();
        }

        // 将矩阵 C 的寄存器变量写入 s_S[4 * Br, Bc]，每个 warp 仅负责 [Br, Bc] 分片，sm_90 之前不支持 stmatrix 指令 (RTX 4090 不支持)
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>((lane >> 2) * Bc       + (lane & 3) * 2))]) = LDST_32_BITS(RC[0]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>(((lane >> 2) + 8) * Bc + (lane & 3) * 2))]) = LDST_32_BITS(RC[1]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>((lane >> 2) * Bc       + (lane & 3) * 2 + 8))]) = LDST_32_BITS(RC[2]);
        LDST_32_BITS(s_S[(warp * Br * Bc) + (swizzle<1, 3, 3>(((lane >> 2) + 8) * Bc + (lane & 3) * 2 + 8))]) = LDST_32_BITS(RC[3]);
        __syncwarp();

        // 对 s_S 求 softmax，每个 warp 单独计算 [Br, Bc] = [16, 16] 矩阵的 softmax，根据 online softmax 先计算 m 和 l
        // 1 个 warp 每次单独处理 2 行，每行 16 个元素，在 warp 内的 16 个线程内部做规约，总共需要处理 Br / 2 = 8 次
#pragma unroll
        for (int j = 0; j < (Br >> 1); ++j) {
            // 读取 2 行数据到 warp
            MD temp_ml = { __half2float(s_S[warp * Br * Bc + j * 32 + lane]) * scale, 1.0f };

            // 每行数据由 16 个线程组成的 group 持有，行内 reduce
            temp_ml = warp_reduce_md<16>(temp_ml);

            // 当前线程处理的行索引
            uint32_t row = warp * Br + j * 2 + (lane >> 4);
            if ((lane & 15) == 0) { // lane = 0 or 16
                row_ml_new[row] = MD_OP()(row_ml_old[row], temp_ml);
            }
            __syncwarp();

            // 行内逐元素更新
            s_S[row * Bc + (lane & 15)] = __float2half(
                __expf(__half2float(s_S[row * Bc + (lane & 15)]) * scale - row_ml_new[row].m)
            );
        }
        
        // 从 s_S[4 * Br, Bc] load 16 × 16 矩阵分片到 RA，每个 warp 仅负责 [Br, Bc] 分片
        uint32_t addr = (warp * Br * Bc) + swizzle<1, 3, 3>((lane & 15) * Bc + (lane >> 4) * 8);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], s_S + addr);

        // 计算 O = PV 矩阵，每次计算尺寸为 16 × 16 × 16
        for (int k = 0; k < d; k += Bd) {
            // 初始化矩阵 C 的寄存器
            RC[0] = RC[1] = RC[2] = RC[3] = 0;

            // 从 s_V[Bc, d] load 16 × 16 矩阵分片到 RB，每个 warp 负责 [Bc, Bd] 分片
            addr = swizzle<3, 3, 4>(k + (lane & 15) * d + (lane >> 4) * 8);
            LDMATRIX_X4_T(RB[0], RB[1], RB[2], RB[3], s_V + addr);

            MMA_M16N8K16_F16F16F16F16(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
            MMA_M16N8K16_F16F16F16F16(RC[2], RC[3], RA[0], RA[1], RA[2], RA[3], RB[2], RB[3], RC[2], RC[3]);

            // 当前线程对应的两组行：row0 / row1
            uint32_t row0 = warp * Br + (lane >> 2);
            uint32_t row1 = row0 + 8;

            // 当前线程对应的列起点（每次处理 half2）
            uint32_t col0 = k + (lane & 3) * 2;

            // s_O 是 [4 * Br, d] 的完整分子累加器
            uint32_t idx0 = warp * Br * d + swizzle<3, 3, 4>((lane >> 2) * d + col0);
            uint32_t idx1 = warp * Br * d + swizzle<3, 3, 4>(((lane >> 2) + 8) * d + col0);
            uint32_t idx2 = warp * Br * d + swizzle<3, 3, 4>((lane >> 2) * d + col0 + 8);
            uint32_t idx3 = warp * Br * d + swizzle<3, 3, 4>(((lane >> 2) + 8) * d + col0 + 8);

            // 4 个 half2 结果
            half2 cur0_h2, cur1_h2, cur2_h2, cur3_h2;
            LDST_32_BITS(cur0_h2) = RC[0];
            LDST_32_BITS(cur1_h2) = RC[1];
            LDST_32_BITS(cur2_h2) = RC[2];
            LDST_32_BITS(cur3_h2) = RC[3];

            float alpha0f = __expf(row_ml_old[row0].m - row_ml_new[row0].m);
            float alpha1f = __expf(row_ml_old[row1].m - row_ml_new[row1].m);

            half2 old0_h2 = LDST_32_BITS(s_O[idx0]);
            half2 old1_h2 = LDST_32_BITS(s_O[idx1]);
            half2 old2_h2 = LDST_32_BITS(s_O[idx2]);
            half2 old3_h2 = LDST_32_BITS(s_O[idx3]);

            float2 old0_f2 = __half22float2(old0_h2);
            float2 old1_f2 = __half22float2(old1_h2);
            float2 old2_f2 = __half22float2(old2_h2);
            float2 old3_f2 = __half22float2(old3_h2);

            float2 cur0_f2 = __half22float2(cur0_h2);
            float2 cur1_f2 = __half22float2(cur1_h2);
            float2 cur2_f2 = __half22float2(cur2_h2);
            float2 cur3_f2 = __half22float2(cur3_h2);

            old0_f2.x = fmaf(alpha0f, old0_f2.x, cur0_f2.x);
            old0_f2.y = fmaf(alpha0f, old0_f2.y, cur0_f2.y);

            old1_f2.x = fmaf(alpha1f, old1_f2.x, cur1_f2.x);
            old1_f2.y = fmaf(alpha1f, old1_f2.y, cur1_f2.y);

            old2_f2.x = fmaf(alpha0f, old2_f2.x, cur2_f2.x);
            old2_f2.y = fmaf(alpha0f, old2_f2.y, cur2_f2.y);

            old3_f2.x = fmaf(alpha1f, old3_f2.x, cur3_f2.x);
            old3_f2.y = fmaf(alpha1f, old3_f2.y, cur3_f2.y);

            LDST_32_BITS(s_O[idx0]) = __float22half2_rn(old0_f2);
            LDST_32_BITS(s_O[idx1]) = __float22half2_rn(old1_f2);
            LDST_32_BITS(s_O[idx2]) = __float22half2_rn(old2_f2);
            LDST_32_BITS(s_O[idx3]) = __float22half2_rn(old3_f2);

            __syncwarp();
        }

        // 更新 row_ml_old
        if (lane < Br) {
            row_ml_old[warp * Br + lane] = row_ml_new[warp * Br + lane];
        }
        __syncthreads();
    }

    for (int i = (lane << 1); i < Br * d; i += (32 << 1)) {
        float2 val_f2 = __half22float2(LDST_32_BITS(s_O[warp * Br * d + (swizzle<3, 3, 4>(i))]));
        float temp = row_ml_new[warp * Br + i / d].d;
        val_f2.x /= temp;
        val_f2.y /= temp;
        LDST_32_BITS(O[qo_offset + warp * Br * d + i]) = __float22half2_rn(val_f2);
    }
}

void launch_flash_attention_v2(half* Q, half* K, half* V, half* O, int batch, int heads, int N, int d, cudaStream_t stream) {
    float scale = rsqrtf(d);
    constexpr int Br = 16;
    constexpr int Bc = 16;
    constexpr int Bd = 16;
    int sram_size = (8 * Br * d + 2 * Bc * d + 4 * Br * Bc) * sizeof(half) + (8 * Br) * sizeof(MD);
    dim3 gridDim(N / (4 * Br), heads, batch);
    dim3 blockDim(128);
    flash_attention_v2_kernel<Br, Bc, Bd><<<gridDim, blockDim, sram_size, stream>>>(Q, K, V, O, N, d, scale);
}
}  // namespace fa2_version7
