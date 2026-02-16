#include <iostream>
#include <cuda_runtime.h>

__global__ void ghist(uint8_t* input, int* hist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        uint8_t in = input[i];
        atomicAdd(&hist[in], 1);
    }
}

__global__ void shist(uint8_t* input, int* hist, int n) {
    // 初始化私有直方图
    __shared__ int shared_hist[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    __syncthreads();
    
    // 构建私有直方图
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        uint8_t in = input[i];
        atomicAdd(&shared_hist[in], 1);
    }
    __syncthreads();

    // 将私有直方图合并到全局直方图
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&hist[i], shared_hist[i]);
    }
}

bool check(int* d_hist1, int* d_hist2, int* hist) {
    int hist1[256], hist2[256];
    cudaMemcpy(hist1, d_hist1, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hist2, d_hist2, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 256; ++i) {
        if (hist1[i] != hist[i] || hist2[i] != hist[i]) {
            std::cout << i << ' ' << hist1[i] << ' ' << hist2[i] << ' ' << hist[i] << "\n";
            return false;
        }
    }
    return true;
}

int main() {
    int M = 10000;
    int N = 10000;
    int size = M * N;
    uint8_t* input = new uint8_t[size];
    int hist[256] = { 0 };
    for (int i = 0; i < size; ++i) {
        input[i] = i % 256;
        hist[input[i]] += 1;
    }

    uint8_t* d_input;
    int* d_hist1;
    int* d_hist2;
    cudaMalloc(&d_input, size * sizeof(uint8_t));
    cudaMalloc(&d_hist1, 256 * sizeof(int));
    cudaMalloc(&d_hist2, 256 * sizeof(int));
    cudaMemset(d_hist1, 0, 256 * sizeof(int));
    cudaMemset(d_hist2, 0, 256 * sizeof(int));
    cudaMemcpy(d_input, input, size * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    dim3 grid_dim(32);
    dim3 block_dim(256);

    for (int i = 0; i < 10; ++i) ghist<<<grid_dim, block_dim>>>(d_input, d_hist1, size);
    cudaMemset(d_hist1, 0, 256 * sizeof(int));
    
    cudaEventRecord(start);
    ghist<<<grid_dim, block_dim>>>(d_input, d_hist1, size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float global_time_ms = 0.0f;
    cudaEventElapsedTime(&global_time_ms, start, end);
    
    for (int i = 0; i < 10; ++i) ghist<<<grid_dim, block_dim>>>(d_input, d_hist2, size);
    cudaMemset(d_hist2, 0, 256 * sizeof(int));

    cudaEventRecord(start);
    shist<<<grid_dim, block_dim>>>(d_input, d_hist2, size);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float shared_time_ms = 0.0f;
    cudaEventElapsedTime(&shared_time_ms, start, end);

    if (check(d_hist1, d_hist2, hist)) {
        std::cout << "success!\n";
    } else {
        std::cout << "failed!\n";
    }
    std::cout << "global kernel execution time: " << global_time_ms << "\n";
    std::cout << "shared kernel execution time: " << shared_time_ms << "\n";

    delete[] input;
    cudaFree(d_input);
    cudaFree(d_hist1);
    cudaFree(d_hist2);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}