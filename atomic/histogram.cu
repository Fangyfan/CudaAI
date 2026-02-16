#include <iostream>
#include <cuda_runtime.h>

__global__ void hist(int8_t* input, int* hist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += gridDim.x * blockDim.x) {
        int8_t in = input[i];
        if (in >= 0 && in < 256) {
            atomicAdd(&hist[in], 1);
        }
    }
}

int main() {
    int M = 3;
    int N = 3;
    int size = M * N;
    int8_t* input = new int8_t[size];
    input[0] = 1; input[1] = 2; input[2] = 3;
    input[3] = 2; input[4] = 3; input[5] = 4;
    input[6] = 3; input[7] = 4; input[8] = 5;

    int8_t* d_input;
    int* d_hist;
    cudaMalloc(&d_input, size * sizeof(int8_t));
    cudaMalloc(&d_hist, 256 * sizeof(int));
    cudaMemset(d_hist, 0, 256 * sizeof(int));

    dim3 block_dim(2);
    dim3 grid_dim(2);
    cudaMemcpy(d_input, input, size * sizeof(int8_t), cudaMemcpyHostToDevice);
    hist<<<grid_dim, block_dim>>>(d_input, d_hist, size);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error: " << err << "\n";
    }

    int h_hist[256];
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 1; i <= 6; ++i) {
        std::cout << "hist[" << i << "] = " << h_hist[i] << "\n";
    }

    delete[] input;
    cudaFree(d_input);
    cudaFree(d_hist);
    return 0;
}