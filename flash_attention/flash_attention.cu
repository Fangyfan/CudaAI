#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <iostream>
#include <assert.h>

constexpr int N = 2048;
constexpr int d = 128;
constexpr int nBlock = N / 128;

namespace naive {
__global__ void naive_self_attention_kernel(float* query, float* key, float* value, float* output, float* score, float* prob, float scale) {
    int tid = (blockIdx.x * blockDim.x + threadIdx.x) * nBlock;
    for (int i = tid; i < tid + nBlock; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < d; ++k) {
                sum += query[i * d + k] * key[j * d + k];
            }
            score[i * N + j] = scale * sum;
        }
    
        float max = -INFINITY;
        for (int j = 0; j < N; ++j) {
            max = fmaxf(max, score[i * N + j]);
        }
        float sump = 0.0f;
        for (int j = 0; j < N; ++j) {
            float temp = expf(score[i * N + j] - max);
            sump += temp;
            prob[i * N + j] = temp;
        }
        for (int j = 0; j < N; ++j) {
            prob[i * N + j] /= sump;
        }
    
        for (int j = 0; j < d; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += prob[i * N + k] * value[k * d + j];
            }
            output[i * d + j] = sum;
        }
    }
}

float* self_attention(float* query, float* key, float* value, float* output, float* score, float* prob) {
    float scale = rsqrtf(d);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    constexpr int block_dim = N / nBlock;
    for (int i = 0; i < 5; ++i) {
        naive_self_attention_kernel<<<1, block_dim>>>(query, key, value, output, score, prob, scale);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        naive_self_attention_kernel<<<1, block_dim>>>(query, key, value, output, score, prob, scale);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);

    std::cout << "Naive self-attention kernel execution time: " << time_ms / 5 << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    float* output_cpu = new float[N * d];
    cudaMemcpy(output_cpu, output, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    return output_cpu;
}
}  // namespace naive

namespace reduce {
template<int BLOCK_DIM_X>
__device__ void softmax_kernel(float* score, float* prob, int d) {
    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM_X>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;

    float max = -INFINITY;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        max = fmaxf(max, score[i]);
    }
    max = BlockReduce(temp).Reduce(max, cub::Max());
    if (threadIdx.x == 0) {
        shared_val = max;
    }
    __syncthreads();
    max = shared_val;

    float sum = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        prob[i] = expf(score[i] - max);
        sum += prob[i];
    }
    sum = BlockReduce(temp).Reduce(sum, cub::Sum());
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;

    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        prob[i] /= sum;
    }
}

template<int BLOCK_DIM_X>
__global__ void self_attention_kernel(float* query, float* key, float* value, float* output, float* score, float* prob, float scale) {
    int bid = blockIdx.x * nBlock;
    for (int i = bid; i < bid + nBlock; ++i) {
        float4* query4 = reinterpret_cast<float4*>(query + i * d);
        for (int j = threadIdx.x; j < N; j += blockDim.x) {
            float4* key4 = reinterpret_cast<float4*>(key + j * d);
            float sum = 0.0f;
            for (int k = 0; k < d / 4; ++k) {
                float4 q4 = query4[k];
                float4 k4 = key4[k];
                sum += (q4.x * k4.x) + (q4.y * k4.y) + (q4.z * k4.z) + (q4.w * k4.w);
            }
            score[i * N + j] = scale * sum;
        }
    
        __syncthreads();
        softmax_kernel<BLOCK_DIM_X>(score + i * N, prob + i * N, N);
        __syncthreads();
    
        for (int j = threadIdx.x; j < d; j += blockDim.x) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += prob[i * N + k] * value[k * d + j];
            }
            output[i * d + j] = sum;
        }
    }
}

float* self_attention(float* query, float* key, float* value, float* output, float* score, float* prob) {
    float scale = rsqrtf(d);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    constexpr int grid_dim = N / nBlock;
    constexpr int block_dim = 128;
    for (int i = 0; i < 5; ++i) {
        self_attention_kernel<block_dim><<<grid_dim, block_dim>>>(query, key, value, output, score, prob, scale);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        self_attention_kernel<block_dim><<<grid_dim, block_dim>>>(query, key, value, output, score, prob, scale);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);

    std::cout << "Reduce self-attention kernel execution time: " << time_ms / 5 << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    float* output_cpu = new float[N * d];
    cudaMemcpy(output_cpu, output, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    return output_cpu;
}
}  // namespace reduce


namespace flash {
constexpr int Br = 8;
constexpr int Bc = 8;
constexpr int Gr = N / Br;
constexpr int Gc = 1;
constexpr int Tc = N / Bc;
constexpr int Dr = d / Br;
constexpr int Dc = d / Bc;
__global__ void flash_attention_v2_kernel(float* query, float* key, float* value, float* output, float scale) {
    __shared__ float Q[Br][d], K[Bc][d], V[Bc][d], O[Br][d];
    __shared__ float S[Br][Bc], Exp[Br][Bc];
    __shared__ float MaxS[Br], SumExp[Br];
    
    int tx = threadIdx.x;  // [0, Bc - 1]
    int ty = threadIdx.y;  // [0, Br - 1]
    int r = blockIdx.y * Br + ty;  // 当前 thread 负责计算全局 query[N][d] 的第 r 行

    // block 内的线程组织为 (Br, Bc)
    // 遍历每个块: Q[Br, Bc] + ... + Q[Br, Bc] = Q[Br, d]
    // 每个 thread 搬运自己负责的 Q[ty][tx], Q[ty][tx + Bc], Q[ty][tx + 2Bc] , ...
    for (int i = 0; i < Dc; ++i) {
        Q[ty][i * Bc + tx] = query[r * d + i * Bc + tx];
        O[ty][i * Bc + tx] = 0;
    }
    if (tx == 0) {
        MaxS[ty] = -INFINITY;
        SumExp[ty] = 0.0f;
    }

    // 每个 block 负责一个 query 的分块 Q[Br, d]
    // 需要遍历第 j = [0, Tc-1] 个 key / value 分块 K[Bc, d], V[Bc, d]
    for (int j = 0; j < Tc; ++j) {
        // block 内的线程组织为 (Br, Bc)
        // 遍历每个块: (K/V)[Bc, Br] + ... + (K/V)[Bc, Br] = (K/V)[Bc, d]
        // 每个 thread 搬运自己负责的 (K/V)[tx][ty], (K/V)[tx][ty + Br], (K/V)[tx][ty + 2Br] , ...
        for (int i = 0; i < Dr; ++i) {
            K[tx][i * Br + ty] = key[(j * Bc + tx) * d + i * Br + ty];
            V[tx][i * Br + ty] = value[(j * Bc + tx) * d + i * Br + ty];
        }
        __syncthreads();

        // 计算点积 S[ty][tx] = QK^T[ty][tx] = Q[ty][0...d-1] * K[tx][0...d-1]
        float sum = 0.0f;
        for (int i = 0; i < d; ++i) {
            sum += Q[ty][i] * K[tx][i];
        }
        S[ty][tx] = scale * sum; // 注意力分数缩放 QK^T / sqrt(d)
        __syncthreads();

        // 计算当前块 S[Br][Bc] 中当前行 S[ty][...] 的局部最大值 localMaxS
        float localMaxS = -INFINITY;
        for (int i = 0; i < Bc; ++i) {
            localMaxS = fmaxf(localMaxS, S[ty][i]);
        }
        // 更新目前的最大值 maxS
        float maxS = fmaxf(MaxS[ty], localMaxS);
        // online softmax 缩放比例: exp(更新前最大值 - 目前最大值)
        float mScale = __expf(MaxS[ty] - maxS);
        // 计算当前块内指数值 P[ty][tx] = exp(S[ty][tx] - m[ty])
        // 注意这里 m[ty] 是当前 ty 行目前的最大值 maxS
        Exp[ty][tx] = __expf(S[ty][tx] - maxS);
        __syncthreads();
        if (tx == 0) MaxS[ty] = maxS;
        
        if (tx == 0) {
            // 计算当前块内 ty 行 Exp[ty][...] 的指数值之和 localSumExp
            float localSumExp = 0.0f;
            for (int i = 0; i < Bc; ++i) {
                localSumExp += Exp[ty][i];
            }
            // 更新前 SumExp[ty] 是以 [更新前最大值] 作为基准
            // 现在要更新基准为 [目前最大值]，因此乘以 mScale
            // 还要加上当前块内指数值之和 localSumExp (本来就是以 [目前最大值] 作为基准)
            SumExp[ty] = mScale * SumExp[ty] + localSumExp;
        }

        // 计算加权部分和 O_i = mScale * O_i-1 + P_i * V_j
        // block 内的线程组织为 (Br, Bc)
        // 遍历每个块: O[Br, Bc] + ... + O[Br, Bc] = O[Br, d]
        // 每个 thread 负责计算 O[ty][tx], O[ty][tx + Bc], O[ty][tx + 2Bc] , ...
        // O[ty][...] = mScale * O[ty][...] + sum_k(P[ty][k] * V[k][...])
        for (int i = 0; i < Dc; ++i) {
            float newO = mScale * O[ty][i * Bc + tx];
            for (int k = 0; k < Bc; ++k) {
                newO += Exp[ty][k] * V[k][i * Bc + tx];
            }
            O[ty][i * Bc + tx] = newO;
        }
        __syncthreads();
    }

    // 每个 thread 负责更新 O[ty][tx], O[ty][tx + Bc], O[ty][tx + 2Bc] , ...
    // 除以 softmax 的分母值 SumExp[ty] = sum(exp(S - max))
    for (int i = 0; i < Dc; ++i) {
        output[r * d + i * Bc + tx] = O[ty][i * Bc + tx] / SumExp[ty];
    }
}

float* self_attention(float* query, float* key, float* value, float* output, float* score, float* prob) {
    float scale = rsqrtf(d);
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    
    dim3 grid_dim(Gc, Gr);
    dim3 block_dim(Bc, Br);

    for (int i = 0; i < 5; ++i) {
        flash_attention_v2_kernel<<<grid_dim, block_dim>>>(query, key, value, output, scale);
    }

    cudaEventRecord(start);
    for (int i = 0; i < 5; ++i) {
        flash_attention_v2_kernel<<<grid_dim, block_dim>>>(query, key, value, output, scale);
    }
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time_ms = 0.0f;
    cudaEventElapsedTime(&time_ms, start, end);

    std::cout << "Flash self-attention kernel execution time: " << time_ms / 5 << " ms" << std::endl;
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    float* output_cpu = new float[N * d];
    cudaMemcpy(output_cpu, output, N * d * sizeof(float), cudaMemcpyDeviceToHost);
    return output_cpu;
}
}  // namespace flash

int main() {
    float* query_cpu = new float[N * d];
    float* key_cpu = new float[N * d];
    float* value_cpu = new float[N * d];
    for (int i = 0; i < N * d; ++i) {
        query_cpu[i] = (float)rand() / RAND_MAX;
        key_cpu[i] = (float)rand() / RAND_MAX;
        value_cpu[i] = (float)rand() / RAND_MAX;
    }

    float* query;
    float* key;
    float* value;
    float* output;
    float* score;
    float* prob;
    cudaMalloc(&query, N * d * sizeof(float));
    cudaMalloc(&key, N * d * sizeof(float));
    cudaMalloc(&value, N * d * sizeof(float));
    cudaMalloc(&output, N * d * sizeof(float));
    cudaMalloc(&score, N * N * sizeof(float));
    cudaMalloc(&prob, N * N * sizeof(float));
    cudaMemcpy(query, query_cpu, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(key, key_cpu, N * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(value, value_cpu, N * d * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemset(output, 0, N * d * sizeof(float));
    float* naive_output = naive::self_attention(query, key, value, output, score, prob);
    cudaMemset(output, 0, N * d * sizeof(float));
    float* reduce_output = reduce::self_attention(query, key, value, output, score, prob);
    cudaMemset(output, 0, N * d * sizeof(float));
    float* flash_output = flash::self_attention(query, key, value, output, score, prob);

    for (int i = 0; i < N * d; ++i) {
        float ref = naive_output[i];
        float val1 = reduce_output[i];
        float val2 = flash_output[i];
        float diff1 = fabsf(ref - val1);
        float diff2 = fabsf(ref - val2);
        float rel_err1 = diff1 / (fabsf(ref) + 1e-7f);
        float rel_err2 = diff2 / (fabsf(ref) + 1e-7f);
        if (rel_err1 > 1e-4 || rel_err2 > 1e-4) {
            std::cout << "[i = " << i << "] : " << naive_output[i] << " vs " << reduce_output[i] << " vs " << flash_output[i] << std::endl;
            std::cout << "Attention kernel failed!" << std::endl;
            return 1;
        }
    }
    std::cout << "Attention kernel successed!" << std::endl;

    cudaFree(query);
    cudaFree(key);
    cudaFree(value);
    cudaFree(output);
    cudaFree(score);
    cudaFree(prob);

    delete[] naive_output;
    delete[] reduce_output;
    delete[] flash_output;
    delete[] query_cpu;
    delete[] key_cpu;
    delete[] value_cpu;
    return 0;
}