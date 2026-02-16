#include <iostream>
#include <cuda_runtime.h>

__global__ void race_condition_kernel(int* data) {
    // 所有线程尝试写入同一个内存位置
    int temp = *data;
    temp = temp + 1;
    *data = temp;
}

int main() {
    int* d_data = 0;
    int h_data = 0;

    cudaMalloc(&d_data, sizeof(int));
    
    cudaMemcpy(d_data, &h_data, sizeof(int), cudaMemcpyHostToDevice);

    race_condition_kernel<<<1024, 256>>>(d_data);
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_data, d_data, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Final value: " << h_data << " (expected " << 1024 * 256 << " if no race condition)\n";

    cudaFree(d_data);

    return 0;
}