#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__global__ void kogge_stone_scan1(float* in, float* out, float* sum, int n) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    T[i] = (offset + i < n) ? in[offset + i] : 0.0f;
    __syncthreads();

    float temp;
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        if (i >= stride) {
            temp = T[i - stride] + T[i];
        }
        __syncthreads();
        if (i >= stride) {
            T[i] = temp;
        }
        __syncthreads();
    }

    if (offset + i < n) out[offset + i] = T[i];
    if (i == blockDim.x - 1) sum[blockIdx.x] = T[blockDim.x - 1];
}

__global__ void kogge_stone_scan2(float* in, float* out, int n) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    T[i] = i < n ? in[i] : 0.0f;
    __syncthreads();

    float sum;
    for (int stride = 1; stride < n; stride <<= 1) {
        if (i >= stride) {
            sum = T[i - stride] + T[i];
        }
        __syncthreads();
        if (i >= stride) {
            T[i] = sum;
        }
        __syncthreads();
    }

    if (i < n) out[i] = T[i];
}

__global__ void kogge_stone_scan11(float* in, float* out, float* sum, int n) {
    extern __shared__ float T[];
    float* source = T;
    float* destination = T + blockDim.x;

    int i = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    source[i] = offset + i < n ? in[offset + i] : 0.0f;
    
    for (int stride = 1; stride < blockDim.x; stride <<= 1) {
        __syncthreads();
        destination[i] = source[i];
        if (i >= stride) {
            destination[i] += source[i - stride];
        }
        float* temp = source;
        source = destination;
        destination = temp;
    }

    if (offset + i < n) out[offset + i] = source[i];
    if (i == blockDim.x - 1) sum[blockIdx.x] = source[blockDim.x - 1];
}

__global__ void kogge_stone_scan22(float* in, float* out, int n) {
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

__global__ void kogge_stone_add_sums(float* out, float* sums, int n) {
    if (blockIdx.x == 0) {
        return;
    }

    int i = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    float add_val = sums[blockIdx.x - 1];

    if (offset + i < n) out[offset + i] += add_val;
}

__global__ void work_efficient_scan1(float* in, float* out, float* sum) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    int m = 2 * blockDim.x;
    int offset = blockIdx.x * m;
    T[2 * i] = in[offset + 2 * i];
    T[2 * i + 1] = in[offset + 2 * i + 1];

    for (int stride = 1; stride < m; stride <<= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (stride <= idx && idx < m) {
            T[idx] += T[idx - stride];
        }
    }

    for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (idx + stride < m) {
            T[idx + stride] += T[idx];
        }
    }
    __syncthreads();

    out[offset + 2 * i] = T[2 * i];
    out[offset + 2 * i + 1] = T[2 * i + 1];
    if (i == blockDim.x - 1) sum[blockIdx.x] = T[m - 1];
}

__global__ void work_efficient_scan2(float* in, float* out, int n) {
    extern __shared__ float T[];

    int i = threadIdx.x;
    T[2 * i] = in[2 * i];
    T[2 * i + 1] = in[2 * i + 1];

    for (int stride = 1; stride < n; stride <<= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (stride <= idx && idx < n) {
            T[idx] += T[idx - stride];
        }
    }

    for (int stride = (blockDim.x >> 1); stride > 0; stride >>= 1) {
        __syncthreads();
        int idx = (i + 1) * (2 * stride) - 1;
        if (idx + stride < n) {
            T[idx + stride] += T[idx];
        }
    }
    __syncthreads();

    out[2 * i] = T[2 * i];
    out[2 * i + 1] = T[2 * i + 1];
}

__global__ void work_efficient_add_sums(float* out, float* sums) {
    if (blockIdx.x == 0) {
        return;
    }

    int i = threadIdx.x;
    int offset = blockIdx.x * (2 * blockDim.x);
    float add_val = sums[blockIdx.x - 1];

    out[offset + 2 * i] += add_val;
    out[offset + 2 * i + 1] += add_val;
}

void benchmark1(std::vector<float>& h_in);
void benchmark2(std::vector<float>& h_in);
int main() {
    constexpr int n = 1024 * 1024;
    constexpr int thread_num = 512;
    constexpr int element_num = 2 * thread_num;
    constexpr int block_num = n / element_num;
    
    std::vector<float> h_in(n);
    std::vector<float> h_ref(n);
    std::vector<float> h_out(n);
    
    for (int i = 0; i < n; ++i) {
        h_in[i] = (float)rand() / RAND_MAX;
        h_ref[i] = h_in[i];
        if (i) h_ref[i] += h_ref[i - 1];
    }
    benchmark1(h_in);
    benchmark2(h_in);
    
    float* d_in;
    float* d_sum;
    float* d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_sum, block_num * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (int i = 0; i < 10; ++i) {
        work_efficient_scan1<<<block_num, thread_num, element_num * sizeof(float)>>>(d_in, d_out, d_sum);
        work_efficient_scan2<<<1, block_num / 2, block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        work_efficient_add_sums<<<block_num, thread_num>>>(d_out, d_sum);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        work_efficient_scan1<<<block_num, thread_num, element_num * sizeof(float)>>>(d_in, d_out, d_sum);
        work_efficient_scan2<<<1, block_num / 2, block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        work_efficient_add_sums<<<block_num, thread_num>>>(d_out, d_sum);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
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
    std::cout << "Work efficient prefix sum kernel execution time: " << milliseconds / 10 << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sum);
    return 0;
}

void benchmark1(std::vector<float>& h_in) {
    constexpr int n = 1024 * 1024;
    constexpr int thread_num = 1024;
    constexpr int block_num = n / thread_num;

    std::vector<float> h_ref(n);
    std::vector<float> h_out(n);

    for (int i = 0; i < n; ++i) {
        h_ref[i] = h_in[i];
        if (i) h_ref[i] += h_ref[i - 1];
    }

    float* d_in;
    float* d_sum;
    float* d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_sum, block_num * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (int i = 0; i < 10; ++i) {
        kogge_stone_scan1<<<block_num, thread_num, thread_num * sizeof(float)>>>(d_in, d_out, d_sum, n);
        kogge_stone_scan2<<<1, block_num, block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        kogge_stone_add_sums<<<block_num, thread_num>>>(d_out, d_sum, n);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        kogge_stone_scan1<<<block_num, thread_num, thread_num * sizeof(float)>>>(d_in, d_out, d_sum, n);
        kogge_stone_scan2<<<1, block_num, block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        kogge_stone_add_sums<<<block_num, thread_num>>>(d_out, d_sum, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        float val = h_out[i];
        float ref = h_ref[i];
        float diff = fabsf(val - ref);
        float rel_err = diff / (fabsf(ref) + 1e-7);
        if (rel_err > 1e-4) {
            std::cout << "Error at index " << i << ": Expected " << ref << ", Got " << val << std::endl;
            return;
        }
    }
    std::cout << "Kogge stone (in-place) prefix sum kernel execution time: " << milliseconds / 10 << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sum);
}

void benchmark2(std::vector<float>& h_in) {
    constexpr int n = 1024 * 1024;
    constexpr int thread_num = 1024;
    constexpr int block_num = n / thread_num;

    std::vector<float> h_ref(n);
    std::vector<float> h_out(n);

    for (int i = 0; i < n; ++i) {
        h_ref[i] = h_in[i];
        if (i) h_ref[i] += h_ref[i - 1];
    }

    float* d_in;
    float* d_sum;
    float* d_out;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_sum, block_num * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    for (int i = 0; i < 10; ++i) {
        kogge_stone_scan11<<<block_num, thread_num, 2 * thread_num * sizeof(float)>>>(d_in, d_out, d_sum, n);
        kogge_stone_scan22<<<1, block_num, 2 * block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        kogge_stone_add_sums<<<block_num, thread_num>>>(d_out, d_sum, n);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 10; ++i) {
        kogge_stone_scan11<<<block_num, thread_num, 2 * thread_num * sizeof(float)>>>(d_in, d_out, d_sum, n);
        kogge_stone_scan22<<<1, block_num, 2 * block_num * sizeof(float)>>>(d_sum, d_sum, block_num);
        kogge_stone_add_sums<<<block_num, thread_num>>>(d_out, d_sum, n);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, end);

    cudaMemcpy(h_out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        float val = h_out[i];
        float ref = h_ref[i];
        float diff = fabsf(val - ref);
        float rel_err = diff / (fabsf(ref) + 1e-7);
        if (rel_err > 1e-4) {
            std::cout << "Error at index " << i << ": Expected " << ref << ", Got " << val << std::endl;
            return;
        }
    }
    std::cout << "Kogge stone (double-buffer) prefix sum kernel execution time: " << milliseconds / 10 << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_sum);
}