#include <cuda_runtime.h>
#include <cmath>
#include <chrono>
#include <random>
#include <vector>
#include <iostream>

void rmsnorm_cpu(float* in, float* weight, float* out, int batch, int size, float eps) {
    for (int i = 0; i < batch; ++i) {
        float* in_ptr = in + i * size;
        float* out_ptr = out + i * size;

        float sum = 0.0f;
        for (int j = 0; j < size; ++j) {
            float val = in_ptr[j];
            sum += val * val;
        }
        
        float scale = rsqrtf(sum / size + eps);
        for (int j = 0; j < size; ++j) {
            out_ptr[j] = in_ptr[j] * weight[j] * scale;
        }
    }
}

__device__ __forceinline__ float warp_reduce(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce(float val) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int warp_num = blockDim.x >> 5;

    val = warp_reduce(val);
    __shared__ float shared_val[32];
    if (lane == 0) {
        shared_val[warp] = val;
    }
    __syncthreads();

    if (warp == 0) {
        val = lane < warp_num ? shared_val[lane] : 0.0f;
        val = warp_reduce(val);
    }
    return val;
}

__global__ void rmsnorm_kernel(float* in, float* weight, float* out, int size, float eps) {
    float* in_ptr = in + blockIdx.x * size;
    float* out_ptr = out + blockIdx.x * size;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        float val = in_ptr[i];
        sum += val * val;
    }
    sum = block_reduce(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    float scale = shared_scale;
    
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        out_ptr[i] = in_ptr[i] * weight[i] * scale;
    }
}

__global__ void rmsnorm_kernel_vector(float* in, float* weight, float* out, int size, float eps) {
    float* in_ptr = in + blockIdx.x * size;
    float* out_ptr = out + blockIdx.x * size;

    int pack_num = size >> 2;
    int tail_off = pack_num << 2;

    float sum = 0.0f;
    float4* in4_ptr = reinterpret_cast<float4*>(in_ptr);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in4_ptr[i];
        sum += (in4.x * in4.x) + (in4.y * in4.y) + (in4.z * in4.z) + (in4.w * in4.w);
    }
    for (int i = tail_off + threadIdx.x; i < size; i += blockDim.x) {
        float val = in_ptr[i];
        sum += val * val;
    }
    sum = block_reduce(sum);

    __shared__ float shared_scale;
    if (threadIdx.x == 0) {
        shared_scale = rsqrtf(sum / size + eps);
    }
    __syncthreads();
    float scale = shared_scale;
    
    float4* wei4_ptr = reinterpret_cast<float4*>(weight);
    float4* out4_ptr = reinterpret_cast<float4*>(out_ptr);
    for (int i = threadIdx.x; i < pack_num; i += blockDim.x) {
        float4 in4 = in4_ptr[i];
        float4 wei4 = wei4_ptr[i];
        out4_ptr[i] = make_float4(
            in4.x * wei4.x * scale, 
            in4.y * wei4.y * scale, 
            in4.z * wei4.z * scale, 
            in4.w * wei4.w * scale
        );
    }
    for (int i = tail_off + threadIdx.x; i < size; i += blockDim.x) {
        out_ptr[i] = in_ptr[i] * weight[i] * scale;
    }
}

int main() {
    int batch = 128;
    int size = 4096;
    float eps = 1e-6f;

    std::vector<float> h_in(batch * size);
    std::vector<float> h_wei(size);
    std::vector<float> h_out_cpu(batch * size);
    std::vector<float> h_out_cu(batch * size);

    std::mt19937 mt(time(nullptr));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < batch * size; ++i) {
        h_in[i] = dist(mt);
        // if (i < 10) std::cout << h_in[i] << "\n";
    }
    for (int i = 0; i < size; ++i) {
        h_wei[i] = dist(mt);
        // if (i < 10) std::cout << h_wei[i] << "\n";
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    rmsnorm_cpu(h_in.data(), h_wei.data(), h_out_cpu.data(), batch, size, eps);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_cpu - start_cpu);
    std::cout << "CPU RMSNorm runtime: " << duration.count() << " ms\n";

    float* d_in;
    float* d_wei;
    float* d_out;
    cudaMalloc(&d_in, batch * size * sizeof(float));
    cudaMalloc(&d_wei, size * sizeof(float));
    cudaMalloc(&d_out, batch * size * sizeof(float));
    cudaMemcpy(d_in, h_in.data(), batch * size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wei, h_wei.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    int warm_up_num = 10;
    for (int i = 0; i < warm_up_num; ++i) {
        rmsnorm_kernel<<<batch, 256>>>(d_in, d_wei, d_out, size, eps);
    }
    cudaEventRecord(start);
    rmsnorm_kernel<<<batch, 256>>>(d_in, d_wei, d_out, size, eps);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float rmsnorm_time_ms = 0.0f;
    cudaEventElapsedTime(&rmsnorm_time_ms, start, end);
    cudaMemcpy(h_out_cu.data(), d_out, batch * size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch * size; ++i) {
        if (fabsf(h_out_cpu[i] - h_out_cu[i]) > 1e-3) {
            std::cout << "rmsnorm kernel failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "GPU RMSNorm runtime: " << rmsnorm_time_ms << " ms\n";

    for (int i = 0; i < warm_up_num; ++i) {
        rmsnorm_kernel_vector<<<batch, 256>>>(d_in, d_wei, d_out, size, eps);
    }
    cudaEventRecord(start);
    rmsnorm_kernel_vector<<<batch, 256>>>(d_in, d_wei, d_out, size, eps);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float rmsnorm_vector_time_ms = 0.0f;
    cudaEventElapsedTime(&rmsnorm_vector_time_ms, start, end);
    cudaMemcpy(h_out_cu.data(), d_out, batch * size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < batch * size; ++i) {
        if (fabsf(h_out_cpu[i] - h_out_cu[i]) > 1e-3) {
            std::cout << "rmsnorm kernel vector failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "GPU RMSNorm Vector runtime: " << rmsnorm_vector_time_ms << " ms\n";

    cudaFree(d_in);
    cudaFree(d_wei);
    cudaFree(d_out);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return 0;
}