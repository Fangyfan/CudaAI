#include <iostream>

__global__ void reduceGmem(int* g_idata, int* g_odata) {
    int tid = threadIdx.x; // 当前线程在当前 block 内的编号

    // 当前 block 负责的数据段的起始地址
    // 第 b 个 block 负责 g_idata[b*512 ... b*512+511]
    int* idata = g_idata + blockIdx.x * blockDim.x;

    if (tid < 256) idata[tid] += idata[tid + 256];
    __syncthreads();
    if (tid < 128) idata[tid] += idata[tid + 128];
    __syncthreads();
    if (tid < 64)  idata[tid] += idata[tid + 64];
    __syncthreads(); // block 内同步屏障

    if (tid < 32) {
        // volatile 的目的：告诉编译器别把 vmem[...] 的访问“缓存到寄存器里不再回读”，传统 reduction 写法里常用来避免优化导致的错误
        volatile int* vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // idata[0] 是当前 block 的和，tid==0 的线程把它加到全局的 *g_odata 上
    if (tid == 0) *g_odata = idata[0];
}

__global__ void reduceSmem(int* g_idata, int* g_odata) {
    // 声明一个 每个 block 私有 的共享内存数组 smem，长度 512
    __shared__ int smem[512];

    int tid = threadIdx.x; // 当前线程在当前 block 内的编号

    // 当前 block 负责的数据段的起始地址
    // 第 b 个 block 负责 g_idata[b*512 ... b*512+511]
    int* idata = g_idata + blockIdx.x * blockDim.x;

    smem[tid] = idata[tid]; // 每个线程把一个输入元素从 global memory 搬到 shared memory
    __syncthreads(); // block 内同步屏障

    if (tid < 256) smem[tid] += smem[tid + 256];
    __syncthreads();
    if (tid < 128) smem[tid] += smem[tid + 128];
    __syncthreads();
    if (tid < 64)  smem[tid] += smem[tid + 64];
    __syncthreads(); // block 内同步屏障

    if (tid < 32) {
        // volatile 的目的：告诉编译器别把 vmem[...] 的访问“缓存到寄存器里不再回读”，传统 reduction 写法里常用来避免优化导致的错误
        volatile int* vmem = smem;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // smem[0] 是当前 block 的和，tid==0 的线程把它加到全局的 *g_odata 上
    if (tid == 0) *g_odata = smem[0];
}

int main() {
    constexpr int n = 10240000;
    int* h_idata = new int[n];
    for (int i = 0; i < n; ++i) h_idata[i] = 1;

    // GPU 设备指针
    int* d_idata1;
    int* d_idata2;
    int* d_odata1;
    int* d_odata2;
    cudaMalloc(&d_idata1, n * sizeof(int));
    cudaMalloc(&d_idata2, n * sizeof(int));
    cudaMalloc(&d_odata1, sizeof(int));
    cudaMalloc(&d_odata2, sizeof(int));

    cudaMemset(d_odata1, 0, sizeof(int)); // 全局和初始化为 0
    cudaMemset(d_odata2, 0, sizeof(int)); // 全局和初始化为 0

    cudaMemcpy(d_idata1, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idata2, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);

    constexpr int thread_num = 512;
    constexpr int block_num = (n + thread_num - 1) / thread_num;

    // Warmup kernels
    for (int i = 0; i < 5; ++i) {
        reduceGmem<<<block_num, thread_num>>>(d_idata1, d_odata1);
        reduceSmem<<<block_num, thread_num>>>(d_idata2, d_odata2);
    }

    cudaMemcpy(d_idata1, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_idata2, h_idata, n * sizeof(int), cudaMemcpyHostToDevice);

    // Event creation: 创建 GPU 事件对象，用来打时间戳
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Measure reduceGmem
    cudaEventRecord(start); // 在默认 stream 上记录开始点
    reduceGmem<<<block_num, thread_num>>>(d_idata1, d_odata1); // 启动 kernel（异步发射到 GPU）
    cudaEventRecord(stop); // 记录结束点（它会排在 kernel 后面）
    cudaEventSynchronize(stop); // 等待 stop 之前的 GPU 工作完成（也就意味着 kernel 执行完了）
    float millisecondsGmem = 0;
    cudaEventElapsedTime(&millisecondsGmem, start, stop); // 计算 start → stop 的毫秒数
    std::cout << "Time for reduceGmem: " << millisecondsGmem << " ms" << std::endl;

    // Measure reduceSmem
    cudaEventRecord(start); // 在默认 stream 上记录开始点
    reduceSmem<<<block_num, thread_num>>>(d_idata2, d_odata2); // 启动 kernel（异步发射到 GPU）
    cudaEventRecord(stop); // 记录结束点（它会排在 kernel 后面）
    cudaEventSynchronize(stop); // 等待 stop 之前的 GPU 工作完成（也就意味着 kernel 执行完了）
    float millisecondsSmem = 0;
    cudaEventElapsedTime(&millisecondsSmem, start, stop); // 计算 start → stop 的毫秒数
    std::cout << "Time for reduceSmem: " << millisecondsSmem << " ms" << std::endl;

    int h_odata1, h_odata2;
    cudaMemcpy(&h_odata1, d_odata1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_odata2, d_odata2, sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "Sum1: " << h_odata1 << std::endl;
    std::cout << "Sum2: " << h_odata2 << std::endl;

    cudaFree(d_idata1);
    cudaFree(d_idata2);
    cudaFree(d_odata1);
    cudaFree(d_odata2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}