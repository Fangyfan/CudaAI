#include <cstdio>

#define BLOCK_SIZE 16  // 每个 block 包含的线程数

static __global__ void vecAdd(int* A, int* B, int* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int res = A[i] + B[i];
        printf("A[i] = %d\n", A[i]);
        C[i] = res;
    }
}

int main() {
    int N = 10;  // 大规模数组
    size_t size = N * sizeof(int);

    int* A = (int*)malloc(size);
    int* B = (int*)malloc(size);
    int* C = (int*)malloc(size);

    // CPU 中去初始化的
    for (int i = 0; i < N; ++i) {
        A[i] = i;
        B[i] = 2 * i;
    }

    int* d_A;
    int* d_B;
    int* d_C;

    // 在 GPU 上分配显存
    // d_A 指向显存的起始位置
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // A 是指向 CPU 内存的指针，d_A 是指向 GPU 内存的指针
    // 将 CPU 内存中的数据拷贝到 GPU 内存中
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    int block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("array size: %d\n", N);
    printf("thread block: %d\n", block_num);
    printf("thread num per block: %d\n", BLOCK_SIZE);
    vecAdd<<<block_num, BLOCK_SIZE>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; ++i) {
        if (C[i] != A[i] + B[i]) {
            printf("Error at index %d: Expected %d, Got %d\n", i, A[i] + B[i], C[i]);
            break;
        }
    }
    printf("Vector addition completed successfully.\n");

    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}