#include <iostream>
#include <cuda_runtime.h>

#define OFFSET(y, x, ld) ((y) * (ld) + (x))
__global__ void naive_transpose_global(float* out, float* in, int ny, int nx) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < nx && y < ny) {
        out[OFFSET(x, y, ny)] = in[OFFSET(y, x, nx)];
    }
}

void call_naiveGmem(float* d_out, float* d_in, int nx, int ny) {
    dim3 block_dim(3, 3);
    dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x, (ny + block_dim.y - 1) / block_dim.y);
    naive_transpose_global<<<grid_dim, block_dim>>>(d_out, d_in, nx, ny);
}

int main() {
    int nx = 4;
    int ny = 4;
    size_t size = nx * ny * sizeof(float);

    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(i % 11);
    }

    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    call_naiveGmem(d_out, d_in, nx, ny);
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            std::cout << h_in[j * nx + i] << " \n"[i == nx - 1];
        }
    }
    std::cout << "--------" << std::endl;
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            std::cout << h_out[j * nx + i] << " \n"[i == nx - 1];
        }
    }
    std::cout << "Matrix transposition completed successfully." << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}