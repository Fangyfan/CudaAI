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

#define BLOCK_DIM_Y 16
#define BLOCK_DIM_X 32
__global__ void transpose_shared(float* out, float* in, int ny, int nx) {
    __shared__ float tile[BLOCK_DIM_Y][BLOCK_DIM_X];

    int x_in = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x_in >= nx || y_in >= ny) return;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * blockDim.x) + y_trans;

    tile[threadIdx.y][threadIdx.x] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans];
}

__global__ void transpose_shared_pad(float* out, float* in, int ny, int nx) {
    constexpr int pad = 1;
    __shared__ float tile[BLOCK_DIM_Y][BLOCK_DIM_X + pad];

    int x_in = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x_in >= nx || y_in >= ny) return;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * blockDim.x) + y_trans;

    tile[threadIdx.y][threadIdx.x] = in[OFFSET(y_in, x_in, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans];
}

__global__ void transpose_shared_unroll_pad(float* out, float* in, int ny, int nx) {
    constexpr int pad = 1;
    __shared__ float tile[BLOCK_DIM_Y][2 * BLOCK_DIM_X + pad];

    int x_in = (blockIdx.x * (blockDim.x * 2)) + threadIdx.x;
    int y_in = (blockIdx.y * blockDim.y) + threadIdx.y;
    if (x_in + BLOCK_DIM_X >= nx || y_in >= ny) return;

    int offset_in_block = threadIdx.y * blockDim.x + threadIdx.x;
    int x_trans = offset_in_block % blockDim.y;
    int y_trans = offset_in_block / blockDim.y;

    int x_out = (blockIdx.y * blockDim.y) + x_trans;
    int y_out = (blockIdx.x * (blockDim.x * 2)) + y_trans;

    tile[threadIdx.y][threadIdx.x] = in[OFFSET(y_in, x_in, nx)];
    tile[threadIdx.y][threadIdx.x + BLOCK_DIM_X] = in[OFFSET(y_in, x_in + BLOCK_DIM_X, nx)];
    __syncthreads();
    out[OFFSET(y_out, x_out, ny)] = tile[x_trans][y_trans];
    out[OFFSET(y_out + BLOCK_DIM_X, x_out, ny)] = tile[x_trans][y_trans + BLOCK_DIM_X];
}

void call_naive_transpose_global(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(32, 32);
    dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x, (ny + block_dim.y - 1) / block_dim.y);
    naive_transpose_global<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x, (ny + block_dim.y - 1) / block_dim.y);
    transpose_shared<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_pad(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_dim((nx + block_dim.x - 1) / block_dim.x, (ny + block_dim.y - 1) / block_dim.y);
    transpose_shared_pad<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
}

void call_transpose_shared_unroll_pad(float* d_out, float* d_in, int ny, int nx) {
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid_dim((nx + (2 * block_dim.x) - 1) / (2 * block_dim.x), (ny + block_dim.y - 1) / block_dim.y);
    transpose_shared_unroll_pad<<<grid_dim, block_dim>>>(d_out, d_in, ny, nx);
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

    int warm_up_times = 10;
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

void test_transpose_shared() {
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

    int warm_up_times = 10;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_pad() {
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

    int warm_up_times = 10;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_pad(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_pad(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

void test_transpose_shared_unroll_pad() {
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

    int warm_up_times = 10;
    for (int i = 0; i < warm_up_times; ++i) {
        call_transpose_shared_unroll_pad(d_out, d_in, ny, nx);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        call_transpose_shared_unroll_pad(d_out, d_in, ny, nx);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);
    
    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (h_out[j * ny + i] != h_in[i * nx + j]) {
                std::cout << "Shared transpose kernel failed!" << std::endl;
                return;
            }
        }
    }
    std::cout << "Shared transpose kernel execution time: " << time_ms / 5 << " ms" << std::endl;

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    test_naive_transpose_global();
    test_transpose_shared();
    test_transpose_shared_pad();
    test_transpose_shared_unroll_pad();
    return 0;
}