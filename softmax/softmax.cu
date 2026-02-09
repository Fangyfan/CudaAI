#include <chrono>
#include <iostream>

void softmax_forward_cpu(float* out, const float* in, int N, int C) {
    for (int i = 0; i < N; ++i) {
        const float *in_row = in + i * C;
        float* out_row = out + i * C;
        // 扫一遍找最大值 max
        float max = -INFINITY;
        for (int j = 0; j < C; ++j) {
            if (max < in_row[j]) {
                max = in_row[j];
            }
        }
        // 再扫一遍：算 exp(x - max) 并累加成 sum
        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            out_row[j] = expf(in_row[j] - max);
            sum += out_row[j];
        }
        // 再扫一遍：把每个指数项除以 sum
        float norm = 1.f / sum;
        for (int j = 0; j < C; ++j) {
            out_row[j] *= norm;
        }
    }
}

// 每一行输入数据分配一个线程块，每个线程块中仅包含 1 个线程
__global__ void softmax_forward_kernel_1(float* out, const float* in, int N, int C) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        const float* in_row = in + i * C;
        float* out_row = out + i * C;
        // 扫一遍找最大值 max
        float max = -INFINITY;
        for (int j = 0; j < C; ++j) {
            if (max < in_row[j]) {
                max = in_row[j];
            }
        }
        // 再扫一遍：算 exp(x - max) 并累加成 sum
        float sum = 0.0f;
        for (int j = 0; j < C; ++j) {
            float temp = expf(in_row[j] - max);
            sum += temp;
            out_row[j] = temp;
        }
        // 再扫一遍：把每个指数项除以 sum
        for (int j = 0; j < C; ++j) {
            out_row[j] /= sum;
        }
    }
}

// 每一行输入数据分配一个 Block，每个 Block 中包含 256 个线程，每个线程负责计算部分和，然后全局规约
__global__ void softmax_forward_kernel_2(float* out, const float* in, int N, int C) {
    // 动态共享内存数组
    extern __shared__ float shared[];
    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;
    
    // 每个线程求自己负责的局部最大值 max(in[t], in[t+B], in[t+2B], ...)
    float max = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        if (max < in_row[i]) {
            max = in_row[i];
        }
    }
    
    // Block 内第 i 个线程的局部最大值存放到 shared[i] 中
    shared[threadIdx.x] = max;
    __syncthreads();

    // 用 shared[0...blockDim-1] 来进行 Block 内树形规约，求 Block 内全局最大值 max(in[0...C-1])
    // [0...15] -> [0...7] -> [0...3] -> [0...1] -> [0]
    for (int stride = (blockDim.x >> 1); stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    max = shared[0]; // 全局最大值为 shared[0]

    // 更新 out[i] = exp(in[i] - max)
    // 每个线程求自己负责的局部和 sum(out[t], out[t+B], out[t+2B], ...)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float temp = expf(in_row[i] - max);
        sum += temp;
        out_row[i] = temp;
    }

    // Block 内第 i 个线程的局部和存放到 shared[i] 中
    shared[threadIdx.x] = sum;
    __syncthreads();

    // 用 shared[0...blockDim-1] 来进行 Block 内树形规约，求 Block 内全局和 sum(out[0...C-1])
    for (int stride = (blockDim.x >> 1); stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    sum = shared[0]; // 全局和为 shared[0]

    // 每个线程求自己负责的局部 softmax 分量 out[t], out[t+B], out[t+2B], ...
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] /= sum;
    }
}

__device__ float warp_reduce_max(float val) {
    for (int offset = (warpSize >> 1); offset; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = (warpSize >> 1); offset; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// 每一行输入数据分配一个 Block，每个 Block 中包含 32 个线程 (1 warp)，每个线程负责计算部分和，然后 warp 规约
__global__ void softmax_forward_kernel_3(float* out, const float* in, int N, int C) {
    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;
    float max = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        if (max < in_row[i]) {
            max = in_row[i];
        }
    }
    max = warp_reduce_max(max);
    max = __shfl_sync(0xFFFFFFFF, max, 0); // 将 max[thread 0] 广播到 warp 内所有线程
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float temp = expf(in_row[i] - max);
        sum += temp;
        out_row[i] = temp;
    }
    sum = warp_reduce_sum(sum);
    sum = __shfl_sync(0xFFFFFFFF, sum, 0); // 将 sum[thread 0] 广播到 warp 内所有线程
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] /= sum;
    }
}

__global__ void softmax_forward_kernel_4(float* out, const float* in, int N, int C) {
    extern __shared__ float shared[];
    const float* in_row = in + blockIdx.x * C;
    float* out_row = out + blockIdx.x * C;

    int warp = threadIdx.x / warpSize;
    int lane = threadIdx.x % warpSize;
    int warp_num = blockDim.x / warpSize;

    float* max_vals = shared;
    float* sum_vals = shared + warp_num;

    // Block 内规约求最大值
    float max = -INFINITY;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        max = fmaxf(max, in_row[i]);
    }
    max = warp_reduce_max(max);
    if (lane == 0) {
        max_vals[warp] = max;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < warp_num; ++i) {
            max = fmax(max, max_vals[i]);
        }
        max_vals[0] = max;
    }
    __syncthreads();
    max = max_vals[0]; // 通过 shared 广播到 Block 内所有线程

    // Block 内规约求和
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        float temp = expf(in_row[i] - max);
        sum += temp;
        out_row[i] = temp;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        sum_vals[warp] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 1; i < warp_num; ++i) {
            sum += sum_vals[i];
        }
        sum_vals[0] = sum;
    }
    __syncthreads();
    sum = sum_vals[0];

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        out_row[i] /= sum;
    }
}

bool compare_results(const float* cpu, const float* gpu, int N, int C, float eps = 1e-3f) {
    for (int i = 0; i < N * C; ++i) {
        if (fabs(cpu[i] - gpu[i]) > eps) {
            std::cout << "Difference at index " << i << ", GPU = " << gpu[i] << " : CPU = " << cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int N = 32;
    constexpr int C = 4096;
    size_t size = N * C * sizeof(float);
    float* in = (float*)malloc(size);
    float* out_cu = (float*)malloc(size);
    float* out_cpu = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < C; ++j) {
            in[i * C + j] = float(i * C + j);
        }
    }

    // Run CPU version and measure time
    auto start_cpu = std::chrono::high_resolution_clock::now();
    softmax_forward_cpu(out_cpu, in, N, C);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;

    // Run GPU version and measure time using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);

    // Warm up
    for (int i = 0; i < 10; ++i) {
        constexpr int block_num = N;
        constexpr int thread_num = 1;
        softmax_forward_kernel_1<<<block_num, thread_num>>>(d_out, d_in, N, C);
    }

    // Launch kernel
    cudaEventRecord(start);
    // constexpr int block_num = N;
    // constexpr int thread_num = 1;
    // softmax_forward_kernel_1<<<block_num, thread_num>>>(d_out, d_in, N, C);

    constexpr int block_num = N;
    constexpr int thread_num = 512;
    constexpr int shared_size = thread_num * sizeof(float);
    softmax_forward_kernel_2<<<block_num, thread_num, shared_size>>>(d_out, d_in, N, C);

    // constexpr int block_num = N;
    // constexpr int thread_num = 32;
    // softmax_forward_kernel_3<<<block_num, thread_num>>>(d_out, d_in, N, C);

    // constexpr int block_num = N;
    // constexpr int thread_num = 512;
    // constexpr int shared_size = (thread_num / 32) * sizeof(float) * 2;
    // softmax_forward_kernel_4<<<block_num, thread_num, shared_size>>>(d_out, d_in, N, C);
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate milliseconds
    float gpu_time_ms = 0;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copy result back to host
    cudaMemcpy(out_cu, d_out, N * C * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Compare results
    bool success = compare_results(out_cpu, out_cu, N, C);
    std::cout << "Results match: " << (success ? "YES" : "NO") << std::endl;

    // Print performance comparison
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << gpu_time_ms << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / (gpu_time_ms)) << "x" << std::endl;

    // Cleanup
    free(in);
    free(out_cpu);
    free(out_cu);

    return 0;
}