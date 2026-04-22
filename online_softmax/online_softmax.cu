#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cub/block/block_reduce.cuh>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

void softmax_forward_cpu(float* out, const float* in, int N, int C) {
    for (int i = 0; i < N; ++i) {
        const float* in_row = in + i * C;
        float* out_row = out + i * C;

        float max_val = -INFINITY;
        for (int j = 0; j < C; ++j) {
            if (max_val < in_row[j]) {
                max_val = in_row[j];
            }
        }

        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            out_row[j] = std::exp(in_row[j] - max_val);
            sum += out_row[j];
        }

        float norm = 1.0f / sum;
        for (int j = 0; j < C; ++j) {
            out_row[j] *= norm;
        }
    }
}

namespace softmax {
template <int BLOCK_DIM>
__global__ void softmax_kernel_1(float* out, const float* in, int C) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float exp_val = __expf(in_row[i] - max_val);
        sum += exp_val;
        out_row[i] = exp_val;
    }
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] /= sum;
    }
}

template <int BLOCK_DIM>
__global__ void softmax_kernel_2(float* out, const float* in, int C) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        max_val = fmaxf(max_val, in_row[i]);
    }
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        sum += __expf(in_row[i] - max_val);
    }
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] = __expf(in_row[i] - max_val) / sum;
    }
}

template <int BLOCK_DIM>
__global__ void softmax_kernel_3(float* out, const float* in, int C) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    const int pack_num = C >> 2;

    float max_val = -INFINITY;
    const float4* in4 = reinterpret_cast<const float4*>(in_row);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        max_val = fmaxf(max_val, v.x);
        max_val = fmaxf(max_val, v.y);
        max_val = fmaxf(max_val, v.z);
        max_val = fmaxf(max_val, v.w);
    }
    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        sum += __expf(v.x - max_val) + __expf(v.y - max_val) + __expf(v.z - max_val) + __expf(v.w - max_val);
    }
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    float4* out4 = reinterpret_cast<float4*>(out_row);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        out4[i] = make_float4(
            __expf(v.x - max_val) / sum,
            __expf(v.y - max_val) / sum,
            __expf(v.z - max_val) / sum,
            __expf(v.w - max_val) / sum
        );
    }
}

template <int BLOCK_DIM>
__global__ void softmax_kernel_4(float* out, const float* in, int C) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    const int pack_num = C >> 2;

    float max_val = -INFINITY;
    float max_val0 = -INFINITY;
    float max_val1 = -INFINITY;
    float max_val2 = -INFINITY;
    float max_val3 = -INFINITY;
    float max_val4 = -INFINITY;
    float max_val5 = -INFINITY;
    const float4* in4 = reinterpret_cast<const float4*>(in_row);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        max_val0 = fmaxf(max_val0, v.x);
        max_val1 = fmaxf(max_val1, v.y);
        max_val2 = fmaxf(max_val2, v.z);
        max_val3 = fmaxf(max_val3, v.w);
    }
    max_val4 = fmaxf(max_val0, max_val1);
    max_val5 = fmaxf(max_val2, max_val3);
    max_val = fmaxf(max_val4, max_val5);

    max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        sum0 += __expf(v.x - max_val);
        sum1 += __expf(v.y - max_val);
        sum2 += __expf(v.z - max_val);
        sum3 += __expf(v.w - max_val);
    }
    sum4 = sum0 + sum1;
    sum5 = sum2 + sum3;
    sum = sum4 + sum5;

    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    float4* out4 = reinterpret_cast<float4*>(out_row);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        out4[i] = make_float4(
            __expf(v.x - max_val) / sum,
            __expf(v.y - max_val) / sum,
            __expf(v.z - max_val) / sum,
            __expf(v.w - max_val) / sum
        );
    }
}

template <int WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE));
    }
    return val;
}

template <int WARP_NUM>
__device__ __forceinline__ float block_reduce_max(float val, float* shared_vals, int lane, int warp) {
    val = warp_reduce_max<32>(val);

    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
        } else {
            val = -INFINITY;
        }
        val = warp_reduce_max<WARP_NUM>(val);
    }
    return val;
}

template <int WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, delta, WARP_SIZE);
    }
    return val;
}

template <int WARP_NUM>
__device__ __forceinline__ float block_reduce_sum(float val, float* shared_vals, int lane, int warp) {
    val = warp_reduce_sum<32>(val);

    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
        } else {
            val = 0.0f;
        }
        val = warp_reduce_sum<WARP_NUM>(val);
    }
    return val;
}

template <int BLOCK_DIM>
__global__ void softmax_kernel_5(float* out, const float* in, int C) {
    constexpr int WARP_NUM = BLOCK_DIM >> 5;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    __shared__ float shared_vals[WARP_NUM];
    __shared__ float shared_val;

    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    const int pack_num = C >> 2;

    float max_val = -INFINITY;
    float max_val0 = -INFINITY;
    float max_val1 = -INFINITY;
    float max_val2 = -INFINITY;
    float max_val3 = -INFINITY;
    float max_val4 = -INFINITY;
    float max_val5 = -INFINITY;
    const float4* in4 = reinterpret_cast<const float4*>(in_row);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        max_val0 = fmaxf(max_val0, v.x);
        max_val1 = fmaxf(max_val1, v.y);
        max_val2 = fmaxf(max_val2, v.z);
        max_val3 = fmaxf(max_val3, v.w);
    }
    max_val4 = fmaxf(max_val0, max_val1);
    max_val5 = fmaxf(max_val2, max_val3);
    max_val = fmaxf(max_val4, max_val5);

    max_val = block_reduce_max<WARP_NUM>(max_val, shared_vals, lane, warp);
    if (threadIdx.x == 0) {
        shared_val = max_val;
    }
    __syncthreads();
    max_val = shared_val;

    float sum = 0.0f;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        sum0 += __expf(v.x - max_val);
        sum1 += __expf(v.y - max_val);
        sum2 += __expf(v.z - max_val);
        sum3 += __expf(v.w - max_val);
    }
    sum4 = sum0 + sum1;
    sum5 = sum2 + sum3;
    sum = sum4 + sum5;

    sum = block_reduce_sum<WARP_NUM>(sum, shared_vals, lane, warp);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    float4* out4 = reinterpret_cast<float4*>(out_row);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        out4[i] = make_float4(
            __expf(v.x - max_val) / sum,
            __expf(v.y - max_val) / sum,
            __expf(v.z - max_val) / sum,
            __expf(v.w - max_val) / sum
        );
    }
}
}  // namespace softmax

namespace online_softmax1 {
struct __align__(8) MD {
    float m;
    float d;
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

template <int BLOCK_DIM>
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int C) {
    input += blockIdx.x * C;
    output += blockIdx.x * C;

    MD md_temp, md_val;
    md_val.m = -INFINITY;
    md_val.d = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        md_temp.m = input[i];
        md_temp.d = 1.0f;
        md_val = MD_OP()(md_val, md_temp);
    }

    using BlockReduce = cub::BlockReduce<MD, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    __shared__ MD shared_md_val;

    md_val = BlockReduce(tempStorage).Reduce(md_val, MD_OP());
    if (threadIdx.x == 0) {
        shared_md_val = md_val;
    }
    __syncthreads();
    md_val = shared_md_val;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        output[i] = __expf(input[i] - md_val.m) / md_val.d;
    }
}

void online_softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C, cudaStream_t stream) {
    dim3 blockDim(512);
    dim3 gridDim(N);
    online_softmax_kernel<512><<<gridDim, blockDim, 0, stream>>>(input, output, C);
}
}  // namespace online_softmax1

namespace online_softmax2 {
struct __align__(8) MD {
    float m;
    float d;
};

struct MD_OP {
    __device__ __forceinline__ MD operator()(const MD& a, const MD& b) const {
        MD res;
        res.m = fmaxf(a.m, b.m);
        res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
        return res;
    }
};

__device__ __forceinline__ MD shfl_down_val(MD val, int delta) {
    MD res;
    res.m = __shfl_down_sync(0xffffffff, val.m, delta);
    res.d = __shfl_down_sync(0xffffffff, val.d, delta);
    return res;
}

__device__ __forceinline__ MD warp_reduce(MD val) {
    auto op = MD_OP();
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
        val = op(val, shfl_down_val(val, delta));
    }
    return val;
}

__device__ __forceinline__ MD block_reduce(MD val) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warp_num = blockDim.x >> 5;

    __shared__ MD shared[32];
    __shared__ MD shared_val;

    val = warp_reduce(val);

    if (lane == 0) {
        shared[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        val = (lane < warp_num) ? shared[lane] : MD{-INFINITY, 0.0f};
        val = warp_reduce(val);
    }

    if (threadIdx.x == 0) {
        shared_val = val;
    }
    __syncthreads();

    val = shared_val;
    return val;
}

template <int BLOCK_DIM>
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int C) {
    input += blockIdx.x * C;
    output += blockIdx.x * C;

    MD md_temp, md_val;
    md_val.m = -INFINITY;
    md_val.d = 0.0f;

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        md_temp.m = input[i];
        md_temp.d = 1.0f;
        md_val = MD_OP()(md_val, md_temp);
    }

    md_val = block_reduce(md_val);

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        output[i] = __expf(input[i] - md_val.m) / md_val.d;
    }
}

void online_softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C, cudaStream_t stream) {
    dim3 blockDim(512);
    dim3 gridDim(N);
    online_softmax_kernel<512><<<gridDim, blockDim, 0, stream>>>(input, output, C);
}
}  // namespace online_softmax2

namespace online_softmax3 {
struct __align__(8) MD {
    float m;
    float d;
};

__device__ __forceinline__ MD merge(const MD& a, const MD& b) {
    MD res;
    res.m = fmaxf(a.m, b.m);
    res.d = a.d * __expf(a.m - res.m) + b.d * __expf(b.m - res.m);
    return res;
}

template <int WARP_SIZE>
__device__ __forceinline__ MD warp_reduce(MD val) {
#pragma unroll
    for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
        val = merge(val, MD{
            __shfl_down_sync(0xffffffff, val.m, delta, WARP_SIZE), 
            __shfl_down_sync(0xffffffff, val.d, delta, WARP_SIZE)
        });
    }
    return val;
}

template <int WARP_NUM>
__device__ __forceinline__ MD block_reduce(MD val) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    __shared__ MD shared_vals[WARP_NUM];
    __shared__ MD shared_val;

    val = warp_reduce<32>(val);

    if (lane == 0) {
        shared_vals[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        if (lane < WARP_NUM) {
            val = shared_vals[lane];
        }
        val = warp_reduce<WARP_NUM>(val);
    }

    if (threadIdx.x == 0) {
        shared_val = val;
    }
    __syncthreads();
    val = shared_val;

    return val;
}

template <int BLOCK_DIM>
__global__ void online_softmax_kernel(const float* __restrict__ input, float* __restrict__ output, int C) {
    input += blockIdx.x * C;
    output += blockIdx.x * C;
    const int pack_num = C >> 2;
    constexpr int WARP_NUM = BLOCK_DIM >> 5;

    MD md_temp[4], md_val;
    md_val.m = -INFINITY;
    md_val.d = 0.0f;

    const float4* in4 = reinterpret_cast<const float4*>(input);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        md_temp[0].m = v.x; md_temp[1].m = v.y;
        md_temp[2].m = v.z; md_temp[3].m = v.w;

        md_temp[0].d = 1.0f; md_temp[1].d = 1.0f;
        md_temp[2].d = 1.0f; md_temp[3].d = 1.0f;

        md_val = merge(md_val, md_temp[0]);
        md_val = merge(md_val, md_temp[1]);
        md_val = merge(md_val, md_temp[2]);
        md_val = merge(md_val, md_temp[3]);
    }
    md_val = block_reduce<WARP_NUM>(md_val);

    float4* out4 = reinterpret_cast<float4*>(output);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 v = in4[i];
        out4[i] = make_float4(
            __expf(v.x - md_val.m) / md_val.d,
            __expf(v.y - md_val.m) / md_val.d,
            __expf(v.z - md_val.m) / md_val.d,
            __expf(v.w - md_val.m) / md_val.d
        );
    }
}

void online_softmax(const float* __restrict__ input, float* __restrict__ output, int N, int C, cudaStream_t stream) {
    dim3 blockDim(512);
    dim3 gridDim(N);
    online_softmax_kernel<512><<<gridDim, blockDim, 0, stream>>>(input, output, C);
}
}  // namespace online_softmax3

struct CompareResult {
    bool passed;
    float max_abs_err;
    int max_err_idx;
};

CompareResult compare_results(const float* ref, const float* test, int N, int C, float eps = 1e-3f) {
    CompareResult result{true, 0.0f, -1};
    for (int i = 0; i < N * C; ++i) {
        float diff = fabsf(ref[i] - test[i]);
        if (diff > result.max_abs_err) {
            result.max_abs_err = diff;
            result.max_err_idx = i;
        }
        if (diff > eps) {
            result.passed = false;
        }
    }
    return result;
}

template <typename LaunchFunc>
float benchmark_kernel(LaunchFunc launch, int warmup_iters, int bench_iters, cudaStream_t stream) {
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

int main() {
    constexpr int N = 128;
    constexpr int C = 8192;
    constexpr int THREADS = 512;
    constexpr int WARMUP_ITERS = 5;
    constexpr int BENCH_ITERS = 5;
    constexpr int KERNEL_NUM = 8;

    static_assert((THREADS % 32) == 0, "THREADS must be a multiple of 32.");

    if ((C & 3) != 0) {
        std::cerr << "C must be a multiple of 4 for float4 kernels." << std::endl;
        return EXIT_FAILURE;
    }

    const size_t elem_num = static_cast<size_t>(N) * C;
    const size_t size = elem_num * sizeof(float);

    float* in = static_cast<float*>(malloc(size));
    float* out_cpu = static_cast<float*>(malloc(size));
    float* out_gpu[KERNEL_NUM] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    if (!in || !out_cpu) {
        std::cerr << "Host malloc failed." << std::endl;
        return EXIT_FAILURE;
    }
    for (int i = 0; i < KERNEL_NUM; ++i) {
        out_gpu[i] = static_cast<float*>(malloc(size));
        if (!out_gpu[i]) {
            std::cerr << "Host malloc failed." << std::endl;
            return EXIT_FAILURE;
        }
    }

    std::srand(123);
    for (size_t i = 0; i < elem_num; ++i) {
        in[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    auto cpu_start = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, in, N, C);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = cpu_end - cpu_start;

    float* d_in = nullptr;
    float* d_out[KERNEL_NUM] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_in), size));
    for (int i = 0; i < KERNEL_NUM; ++i) {
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&d_out[i]), size));
    }

    CHECK_CUDA(cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int block_num = N;
    constexpr int thread_num = THREADS;

    auto launch_kernel_1 = [&]() {
        softmax::softmax_kernel_1<thread_num><<<block_num, thread_num, 0, stream>>>(d_out[0], d_in, C);
    };

    auto launch_kernel_2 = [&]() {
        softmax::softmax_kernel_2<thread_num><<<block_num, thread_num, 0, stream>>>(d_out[1], d_in, C);
    };

    auto launch_kernel_3 = [&]() {
        softmax::softmax_kernel_3<thread_num><<<block_num, thread_num, 0, stream>>>(d_out[2], d_in, C);
    };

    auto launch_kernel_4 = [&]() {
        softmax::softmax_kernel_4<thread_num><<<block_num, thread_num, 0, stream>>>(d_out[3], d_in, C);
    };

    auto launch_kernel_5 = [&]() {
        softmax::softmax_kernel_5<thread_num><<<block_num, thread_num, 0, stream>>>(d_out[4], d_in, C);
    };

    auto launch_kernel_6 = [&]() {
        online_softmax1::online_softmax_kernel<thread_num><<<block_num, thread_num, 0, stream>>>(d_in, d_out[5], C);
    };

    auto launch_kernel_7 = [&]() {
        online_softmax2::online_softmax_kernel<thread_num><<<block_num, thread_num, 0, stream>>>(d_in, d_out[6], C);
    };

    auto launch_kernel_8 = [&]() {
        online_softmax3::online_softmax_kernel<thread_num><<<block_num, thread_num, 0, stream>>>(d_in, d_out[7], C);
    };

    float kernel_ms[KERNEL_NUM];
    kernel_ms[0] = benchmark_kernel(launch_kernel_1, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[1] = benchmark_kernel(launch_kernel_2, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[2] = benchmark_kernel(launch_kernel_3, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[3] = benchmark_kernel(launch_kernel_4, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[4] = benchmark_kernel(launch_kernel_5, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[5] = benchmark_kernel(launch_kernel_6, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[6] = benchmark_kernel(launch_kernel_7, WARMUP_ITERS, BENCH_ITERS, stream);
    kernel_ms[7] = benchmark_kernel(launch_kernel_8, WARMUP_ITERS, BENCH_ITERS, stream);

    for (int i = 0; i < KERNEL_NUM; ++i) {
        CHECK_CUDA(cudaMemcpy(out_gpu[i], d_out[i], size, cudaMemcpyDeviceToHost));
    }

    CompareResult cmp[KERNEL_NUM];
    for (int i = 0; i < KERNEL_NUM; ++i) {
        cmp[i] = compare_results(out_cpu, out_gpu[i], N, C);
    }

    const char* kernel_names[KERNEL_NUM] = {
        "softmax_kernel_1",
        "softmax_kernel_2",
        "softmax_kernel_3",
        "softmax_kernel_4",
        "softmax_kernel_5",
        "online_softmax1",
        "online_softmax2",
        "online_softmax3"
    };

    int best_idx = 0;
    for (int i = 1; i < KERNEL_NUM; ++i) {
        if (kernel_ms[i] < kernel_ms[best_idx]) {
            best_idx = i;
        }
    }

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "================ Benchmark Result ================\n";
    std::cout << "Shape: N = " << N << ", C = " << C << "\n";
    std::cout << "Threads per block: " << THREADS << "\n";
    std::cout << "Warmup iterations: " << WARMUP_ITERS << "\n";
    std::cout << "Benchmark iterations: " << BENCH_ITERS << "\n";
    std::cout << "CPU reference time: " << cpu_time.count() << " ms\n\n";

    std::cout << std::left
              << std::setw(20) << "Kernel"
              << std::setw(14) << "avg_ms"
              << std::setw(14) << "vs_best"
              << std::setw(12) << "vs_cpu"
              << std::setw(10) << "Result"
              << std::setw(14) << "max_abs_err"
              << std::setw(12) << "max_err_idx"
              << "\n";

    for (int i = 0; i < KERNEL_NUM; ++i) {
        std::cout << std::left
                  << std::setw(20) << kernel_names[i]
                  << std::setw(14) << kernel_ms[i]
                  << std::setw(14) << (kernel_ms[best_idx] / kernel_ms[i])
                  << std::setw(12) << (cpu_time.count() / kernel_ms[i])
                  << std::setw(10) << (cmp[i].passed ? "PASS" : "FAIL")
                  << std::setw(14) << cmp[i].max_abs_err
                  << std::setw(12) << cmp[i].max_err_idx
                  << "\n";
    }

    std::cout << "\nBest kernel: " << kernel_names[best_idx]
              << " (" << kernel_ms[best_idx] << " ms)\n";
    std::cout << "==================================================\n";

    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_in));
    for (int i = 0; i < KERNEL_NUM; ++i) {
        CHECK_CUDA(cudaFree(d_out[i]));
    }

    free(in);
    free(out_cpu);
    for (int i = 0; i < KERNEL_NUM; ++i) {
        free(out_gpu[i]);
    }

    return 0;
}
