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

#define BLOCK_DIM_Y_16 16
#define BLOCK_DIM_X_32 32
__global__ void transpose_shared_16_32(float* out, float* in, int ny, int nx) {
    __shared__ float tile[BLOCK_DIM_Y_16][BLOCK_DIM_X_32];

    int x_in = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * blockDim.x) + y_trans;

    tile[threadIdx.y][threadIdx.x] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans];
}

__global__ void transpose_shared_16_32_padding(float* out, float* in, int ny, int nx) {
    __shared__ float tile[BLOCK_DIM_Y_16][BLOCK_DIM_X_32 + 1];

    int x_in = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * blockDim.x) + y_trans;

    tile[threadIdx.y][threadIdx.x] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans];
}

__global__ void transpose_shared_16_32_swizzle(float* out, const float* in, int ny, int nx) {
    __shared__ float tile[BLOCK_DIM_Y_16][BLOCK_DIM_X_32];

    int x_in = blockIdx.x * blockDim.x + threadIdx.x;
    int y_in = blockIdx.y * blockDim.y + threadIdx.y;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = blockIdx.y * blockDim.y + x_trans;
    int y_out = blockIdx.x * blockDim.x + y_trans;

    tile[threadIdx.y][threadIdx.x ^ (threadIdx.y << 1)] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans ^ (x_trans << 1)];
}

#define BLOCK_DIM_Y_32 32
#define BLOCK_DIM_X_16 16
__global__ void transpose_shared_32_16_padding(float* out, float* in, int ny, int nx) {
    __shared__ float tile[BLOCK_DIM_Y_32][BLOCK_DIM_X_16 + 1];

    int x_in = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * blockDim.x) + y_trans;

    tile[threadIdx.y][threadIdx.x] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans];
}

__global__ void transpose_shared_32_16_swizzle(float* out, float* in, int ny, int nx) {
    __shared__ float tile[BLOCK_DIM_Y_32][BLOCK_DIM_X_16];

    int x_in = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * blockDim.x) + y_trans;

    tile[threadIdx.y][threadIdx.x ^ (threadIdx.y >> 1)] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans ^ (x_trans >> 1)];
}

void call_naive_transpose_global(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(32, 32);
    dim3 grid_dim(nx / block_dim.x, ny / block_dim.y);
    naive_transpose_global<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_16_32(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X_32, BLOCK_DIM_Y_16);
    dim3 grid_dim(nx / block_dim.x, ny / block_dim.y);
    transpose_shared_16_32<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_16_32_padding(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X_32, BLOCK_DIM_Y_16);
    dim3 grid_dim(nx / block_dim.x, ny / block_dim.y);
    transpose_shared_16_32_padding<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_16_32_swizzle(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X_32, BLOCK_DIM_Y_16);
    dim3 grid_dim(nx / block_dim.x, ny / block_dim.y);
    transpose_shared_16_32_swizzle<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_32_16_padding(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X_16, BLOCK_DIM_Y_32);
    dim3 grid_dim(nx / block_dim.x, ny / block_dim.y);
    transpose_shared_32_16_padding<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_32_16_swizzle(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X_16, BLOCK_DIM_Y_32);
    dim3 grid_dim(nx / block_dim.x, ny / block_dim.y);
    transpose_shared_32_16_swizzle<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void test_naive_transpose_global() {
    int ny = 4096;
    int nx = 4096;
    size_t size = ny * nx * sizeof(float);
    
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(rand());
    }
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_times = 5;
    for (int i = 0; i < warm_up_times; ++i) {
        call_naive_transpose_global(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_naive_transpose_global(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Naive transpose kernel failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Naive transpose kernel execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_16_32() {
    int ny = 4096;
    int nx = 4096;
    size_t size = ny * nx * sizeof(float);
    
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(rand());
    }
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_times = 5;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_16_32(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_16_32(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel 16 × 32 failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel 16 × 32 execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_16_32_padding() {
    int ny = 4096;
    int nx = 4096;
    size_t size = ny * nx * sizeof(float);
    
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(rand());
    }
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_times = 5;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_16_32_padding(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_16_32_padding(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel 16 × 32 (padding) failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel 16 × 32 (padding) execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_16_32_swizzle() {
    int ny = 4096;
    int nx = 4096;
    size_t size = ny * nx * sizeof(float);
    
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(rand());
    }
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_times = 5;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_16_32_swizzle(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_16_32_swizzle(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel 16 × 32 (swizzle) failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel 16 × 32 (swizzle) execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_32_16_padding() {
    int ny = 4096;
    int nx = 4096;
    size_t size = ny * nx * sizeof(float);
    
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(rand());
    }
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_times = 5;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_32_16_padding(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_32_16_padding(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel 32 × 16 (padding) failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel 32 × 16 (padding) execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_32_16_swizzle() {
    int ny = 4096;
    int nx = 4096;
    size_t size = ny * nx * sizeof(float);
    
    float* h_in = (float*)malloc(size);
    float* h_out = (float*)malloc(size);
    
    for (int i = 0; i < ny * nx; ++i) {
        h_in[i] = float(rand());
    }
    
    float* d_in;
    float* d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_times = 5;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_32_16_swizzle(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_32_16_swizzle(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel 32 × 16 (swizzle) failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel 32 × 16 (swizzle) execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    test_naive_transpose_global();
    test_transpose_shared_16_32();
    test_transpose_shared_16_32_padding();
    test_transpose_shared_16_32_swizzle();
    test_transpose_shared_32_16_padding();
    test_transpose_shared_32_16_swizzle();
    return 0;
}