#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>
#include <random>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)             \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)


namespace Topk_Quick_Select {
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
int partition(int* a, int left, int right) {
    int i = left + rng() % (right - left + 1);
    int flag = a[i];
    std::swap(a[i], a[left]);
    i = left + 1;
    int j = right;
    while (true) {
        while (i <= j && a[i] > flag) ++i; // a[i] <= flag
        while (i <= j && a[j] < flag) --j; // a[j] >= flag
        if (i >= j) break;
        std::swap(a[i], a[j]);
        ++i;
        --j;
    }
    std::swap(a[left], a[j]);
    return j;
}

void quick_select(int* a, int left, int right, int k) {
    while (true) {
        int p = partition(a, left, right); // left <= p <= right
        if (p == k - 1) return;
        if (p < k - 1) {
            left = p + 1;
        } else {
            right = p - 1;
        }
    }
}

void topk_quick_select(int* input, int* temp, int* output, int n, int k) {
    // 复制数据以避免修改原数组
    memcpy(temp, input, n * sizeof(int));

    // 使用快速选择找到前 k 个最大元素
    quick_select(temp, 0, n - 1, k);

    // 复制前 k 个元素到输出
    memcpy(output, temp, k * sizeof(int));

    // 对结果进行排序（降序）
    std::sort(output, output + k, std::greater<int>());
}
}

namespace Topk_Priority_Queue {
void topk_priority_queue(int* input, int* output, int n, int k) {
    // 使用最小堆维护前 k 个最大元素
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;

    // 遍历所有元素
    for (int i = 0; i < n; ++i) {
        if (min_heap.size() < k) {
            // 堆未满，直接插入
            min_heap.push(input[i]);
        } else if (input[i] > min_heap.top()) {
            // 当前元素大于堆顶，替换堆顶元素
            min_heap.pop();
            min_heap.push(input[i]);
        }
    }

    // 从堆中提取结果（需要反转顺序，因为堆顶是最小的）
    for (int i = k - 1; i >= 0; --i) {
        output[i] = min_heap.top();
        min_heap.pop();
    }
}
}

namespace Topk_One_Block_Reduce {
template <int TOPK>
__device__ __forceinline__ void update_topk(int* topk, int val) {
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
        if (topk[i] < val) {
            int temp = topk[i];
            topk[i] = val;
            val = temp;
        }
    }
}

template <int TOPK, int BLOCK_DIM>
__global__ void topk_kernel(int* input, int* output, int n) {
    // 使用共享内存存储每个块的中间结果 (Block 内 topk)
    __shared__ int shared_topk[BLOCK_DIM * TOPK];
    int local_topk[TOPK];

    // 当前线程局部 topk 初始化
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
        local_topk[i] = INT32_MIN;
    }

    // Block 内跨步循环处理数据
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        update_topk<TOPK>(local_topk, input[i]);
    }

    // 将各线程结果存入共享内存
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
        shared_topk[TOPK * threadIdx.x + i] = local_topk[i];
    }
    __syncthreads();

    // 块内规约：合并块内所有线程的 topk 结果
#pragma unroll
    for (int stride = (BLOCK_DIM >> 1); stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
#pragma unroll
            for (int i = 0; i < TOPK; ++i) {
                update_topk<TOPK>(local_topk, shared_topk[TOPK * (threadIdx.x + stride) + i]);
            }
        }
        __syncthreads();

        if (threadIdx.x < stride) {
#pragma unroll
            for (int i = 0; i < TOPK; ++i) {
                shared_topk[TOPK * threadIdx.x + i] = local_topk[i];
            }
        }
        __syncthreads();
    }

    // thread 0 将最终 topk 结果写入全局内存
    if (threadIdx.x == 0) {
#pragma unroll
        for (int i = 0; i < TOPK; ++i) {
            output[i] = local_topk[i];
        }
    }
}

template <int TOPK>
void launch_topk(int* input, int* output, int n, cudaStream_t stream) {
    dim3 gridDim(1);
    dim3 blockDim(512);
    topk_kernel<TOPK, 512><<<gridDim, blockDim, 0, stream>>>(input, output, n);
}
}

namespace bench_utils {

struct CompareResult {
    bool pass = true;
    int first_bad_idx = -1;
    int ref_val = 0;
    int test_val = 0;
};

struct Report {
    std::string name;
    double avg_ms = 0.0;
    bool pass = true;
    int first_bad_idx = -1;
    int ref_val = 0;
    int test_val = 0;
};

inline void cpu_reference_topk(const int* input, int* output, int n, int k) {
    // 作为 baseline / 真值：
    // partial_sort_copy 会直接得到“按降序排列”的 top-k，且不修改 input
    std::partial_sort_copy(
        input, input + n,
        output, output + k,
        std::greater<int>()
    );
}

inline CompareResult compare_topk_exact(const int* ref, const int* test, int k) {
    CompareResult r;
    for (int i = 0; i < k; ++i) {
        if (ref[i] != test[i]) {
            r.pass = false;
            r.first_bad_idx = i;
            r.ref_val = ref[i];
            r.test_val = test[i];
            return r;
        }
    }
    return r;
}

template <typename Fn>
double benchmark_cpu_ms(Fn&& fn, int warmup_iters, int bench_iters) {
    for (int i = 0; i < warmup_iters; ++i) {
        fn();
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < bench_iters; ++i) {
        fn();
    }
    auto end = std::chrono::steady_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return total_ms / bench_iters;
}

template <typename PrepareFn, typename LaunchFn>
float benchmark_gpu_ms(PrepareFn&& prepare,
                       LaunchFn&& launch,
                       int warmup_iters,
                       int bench_iters,
                       cudaStream_t stream) {
    for (int i = 0; i < warmup_iters; ++i) {
        prepare();
        launch();
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float total_ms = 0.0f;
    for (int i = 0; i < bench_iters; ++i) {
        prepare();

        CHECK_CUDA(cudaEventRecord(start, stream));
        launch();
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaEventRecord(stop, stream));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float iter_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&iter_ms, start, stop));
        total_ms += iter_ms;
    }

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return total_ms / bench_iters;
}

inline void print_report_table(const std::vector<Report>& reports, double baseline_ms) {
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "==================== TopK Benchmark ====================\n";
    std::cout << std::left
              << std::setw(28) << "name"
              << std::setw(14) << "avg_ms"
              << std::setw(18) << "speedup_vs_base"
              << std::setw(10) << "correct"
              << "detail\n";

    for (const auto& r : reports) {
        std::cout << std::left
                  << std::setw(28) << r.name
                  << std::setw(14) << r.avg_ms
                  << std::setw(18) << (baseline_ms / r.avg_ms)
                  << std::setw(10) << (r.pass ? "PASS" : "FAIL");

        if (!r.pass) {
            std::cout << "first_bad_idx=" << r.first_bad_idx
                      << ", ref=" << r.ref_val
                      << ", got=" << r.test_val;
        }
        std::cout << "\n";
    }
    std::cout << "========================================================\n";
}

}  // namespace bench_utils

int main() {
    constexpr int N = 10000000;
    constexpr int TOPK = 20;
    constexpr int warmup__iters = 10;
    constexpr int bench__iters  = 20;

    // -----------------------------
    // 1) 输入数据：保留你原文件中的初始化逻辑，不要改
    // -----------------------------
    std::vector<int> h_input(N);

    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, INT_MAX);
    for (int i = 0; i < N; ++i) h_input[i] = dist(rng);

    // -----------------------------
    // 2) host 侧缓冲区
    // -----------------------------
    std::vector<int> h_ref(TOPK);
    std::vector<int> h_out_baseline(TOPK);
    std::vector<int> h_out_quick(TOPK);
    std::vector<int> h_out_pq(TOPK);
    std::vector<int> h_out_gpu(TOPK);
    std::vector<int> h_temp_quick(N);   // quick_select 复用 temp，避免反复分配

    // -----------------------------
    // 3) device 侧缓冲区
    // -----------------------------
    int* d_input = nullptr;
    int* d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input,  static_cast<size_t>(N)    * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, static_cast<size_t>(TOPK) * sizeof(int)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // H2D 放在计时区间外，只测 kernel 本体
    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          static_cast<size_t>(N) * sizeof(int),
                          cudaMemcpyHostToDevice));

    // -----------------------------
    // 4) 先生成真值 reference（不计时）
    // -----------------------------
    bench_utils::cpu_reference_topk(h_input.data(), h_ref.data(), N, TOPK);

    std::vector<bench_utils::Report> reports;

    // -----------------------------
    // 5) baseline: CPU reference
    // -----------------------------
    double baseline_ms = bench_utils::benchmark_cpu_ms(
        [&]() {
            bench_utils::cpu_reference_topk(h_input.data(), h_out_baseline.data(), N, TOPK);
        },
        warmup__iters,
        bench__iters
    );

    auto baseline_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_baseline.data(), TOPK);
    reports.push_back({
        "baseline_cpu",
        baseline_ms,
        baseline_cmp.pass,
        baseline_cmp.first_bad_idx,
        baseline_cmp.ref_val,
        baseline_cmp.test_val
    });

    // -----------------------------
    // 6) CPU: Quick Select
    // -----------------------------
    double quick_ms = bench_utils::benchmark_cpu_ms(
        [&]() {
            Topk_Quick_Select::topk_quick_select(
                h_input.data(),
                h_temp_quick.data(),
                h_out_quick.data(),
                N,
                TOPK
            );
        },
        warmup__iters,
        bench__iters
    );

    auto quick_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_quick.data(), TOPK);
    reports.push_back({
        "Topk_Quick_Select",
        quick_ms,
        quick_cmp.pass,
        quick_cmp.first_bad_idx,
        quick_cmp.ref_val,
        quick_cmp.test_val
    });

    // -----------------------------
    // 7) CPU: Priority Queue
    // -----------------------------
    double pq_ms = bench_utils::benchmark_cpu_ms(
        [&]() {
            Topk_Priority_Queue::topk_priority_queue(
                h_input.data(),
                h_out_pq.data(),
                N,
                TOPK
            );
        },
        warmup__iters,
        bench__iters
    );

    auto pq_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_pq.data(), TOPK);
    reports.push_back({
        "Topk_Priority_Queue",
        pq_ms,
        pq_cmp.pass,
        pq_cmp.first_bad_idx,
        pq_cmp.ref_val,
        pq_cmp.test_val
    });

    // -----------------------------
    // 8) GPU: One Block Reduce
    //    只测 kernel 时间，不含 H2D / D2H
    // -----------------------------
    float gpu_ms = bench_utils::benchmark_gpu_ms(
        [&]() {
            // 这个 kernel 每次都会覆盖 output，因此这里无需额外 prepare
        },
        [&]() {
            Topk_One_Block_Reduce::topk_kernel<TOPK, 512>
                <<<1, 512, 0, stream>>>(d_input, d_output, N);
        },
        warmup__iters,
        bench__iters,
        stream
    );

    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(),
                          d_output,
                          static_cast<size_t>(TOPK) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto gpu_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_gpu.data(), TOPK);
    reports.push_back({
        "Topk_One_Block_Reduce",
        static_cast<double>(gpu_ms),
        gpu_cmp.pass,
        gpu_cmp.first_bad_idx,
        gpu_cmp.ref_val,
        gpu_cmp.test_val
    });

    // -----------------------------
    // 9) 打印结果
    // -----------------------------
    std::cout << "N = " << N
              << ", TOPK = " << TOPK
              << ", warmup = " << warmup__iters
              << ", bench = " << bench__iters << "\n";
    std::cout << "(GPU time is kernel-only; H2D / D2H are excluded)\n\n";

    bench_utils::print_report_table(reports, baseline_ms);

    // -----------------------------
    // 10) 资源释放
    // -----------------------------
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaStreamDestroy(stream));

    return 0;
}