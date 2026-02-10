#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <vector>

constexpr int N = 1024 * 1024;
constexpr int thread_num = 1024;
constexpr int block_num = N / (2 * thread_num);

__global__ void reduce_v2(float* g_idata, float* g_odata) {
    __shared__ float shared[thread_num];

    int i = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    shared[threadIdx.x] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    for (int stride = (blockDim.x >> 1); stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        g_odata[blockIdx.x] = shared[0];
    }
}

float reduce_cpu(const std::vector<float>& data) {
    float sum = 0.0f;
    for (float val : data) {
        sum += val;
    }
    return sum;
}

int main() {
    std::vector<float> h_data(N);
    for (int i = 0; i < N; ++i) {
        h_data[i] = 1.0f;
    }

    // CPU 计时开始
    auto start_cpu = std::chrono::high_resolution_clock::now();
    float out_cpu = reduce_cpu(h_data);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_time = end_cpu - start_cpu;
    // CPU 计时结束

    float* d_data;
    float* d_out;
    float* d_final_out;
    float out_cu;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_out, block_num * sizeof(float));
    cudaMalloc(&d_final_out, sizeof(float));
    cudaMemcpy(d_data, h_data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < 10; ++i) {
        reduce_v2<<<block_num, thread_num>>>(d_data, d_out);
        reduce_v2<<<1, block_num / 2>>>(d_out, d_final_out);
    }

    // GPU 计时开始 (CUDA Events)
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduce_v2<<<block_num, thread_num>>>(d_data, d_out);
    reduce_v2<<<1, block_num / 2>>>(d_out, d_final_out);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    // GPU 计时结束

    cudaMemcpy(&out_cu, d_final_out, sizeof(float), cudaMemcpyDeviceToHost);
    
    if (fabs(out_cpu - out_cu) < 1e-5) {
        std::cout << "Result verified successfully!" << std::endl;
    } else {
        std::cout << "Result verification failed!" << std::endl;
    }
    std::cout << "CPU result: " << out_cpu << std::endl;
    std::cout << "GPU result: " << out_cu << std::endl;
    std::cout << "CPU time: " << cpu_time.count() << " ms" << std::endl;
    std::cout << "GPU time: " << milliseconds << " ms" << std::endl;
    std::cout << "Speedup: " << (cpu_time.count() / milliseconds) << "x" << std::endl;

    cudaFree(d_data);
    cudaFree(d_out);
    cudaFree(d_final_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}