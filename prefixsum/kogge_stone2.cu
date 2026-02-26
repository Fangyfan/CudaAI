#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void kogge_stone_scan(float* in, float* out, int n) {
    extern __shared__ float T[];
    float* source = T;
    float* destination = T + n;

    int i = threadIdx.x;
    source[i] = i < n ? in[i] : 0.0f;
    
    for (int stride = 1; stride < n; stride <<= 1) {
        __syncthreads();
        destination[i] = source[i];
        if (i >= stride) {
            destination[i] += source[i - stride];
        }
        float* temp = source;
        source = destination;
        destination = temp;
    }

    if (i < n) out[i] = source[i];
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);
    std::vector<float> h_in(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_out(n);

    for (int i = 0; i < n; ++i) {
        h_in[i] = (float)rand() / RAND_MAX;
        h_ref[i] = h_in[i];
        if (i) h_ref[i] += h_ref[i - 1];
    }

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in.data(), size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (int i = 0; i < 10; ++i) {
        kogge_stone_scan<<<1, n, 2 * size>>>(d_in, d_out, n);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        kogge_stone_scan<<<1, n, 2 * size>>>(d_in, d_out, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    cudaMemcpy(h_out.data(), d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        float val = h_out[i];
        float ref = h_ref[i];
        float diff = fabsf(val - ref);
        float rel_err = diff / (fabsf(ref) + 1e-7);
        if (rel_err > 1e-4) {
            std::cout << "Error at index " << i << ": Expected " << ref << ", Got " << val << std::endl;
            return 1;
        }
    }
    std::cout << "Prefix sum kernel execution time: " << milliseconds / 10 << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}