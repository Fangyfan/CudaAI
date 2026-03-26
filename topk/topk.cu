#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
#include <string>
#include <vector>

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
    memcpy(temp, input, static_cast<size_t>(n) * sizeof(int));
    quick_select(temp, 0, n - 1, k);
    memcpy(output, temp, static_cast<size_t>(k) * sizeof(int));
    std::sort(output, output + k, std::greater<int>());
}
} // namespace Topk_Quick_Select

namespace Topk_Priority_Queue {
void topk_priority_queue(int* input, int* output, int n, int k) {
    std::priority_queue<int, std::vector<int>, std::greater<int>> min_heap;

    for (int i = 0; i < n; ++i) {
        if (static_cast<int>(min_heap.size()) < k) {
            min_heap.push(input[i]);
        } else if (input[i] > min_heap.top()) {
            min_heap.pop();
            min_heap.push(input[i]);
        }
    }

    for (int i = k - 1; i >= 0; --i) {
        output[i] = min_heap.top();
        min_heap.pop();
    }
}
} // namespace Topk_Priority_Queue

namespace Topk_One_Block_Reduce {
template <int TOPK>
__device__ __forceinline__ void update_topk(int* topk, int val) {
    if (topk[TOPK - 1] >= val) {
        return;
    }

    for (int i = TOPK - 2; i >= 0; i--) {
        if (val > topk[i]) {
            topk[i + 1] = topk[i];
        } else {
            topk[i + 1] = val;
            return;
        }
    }
    topk[0] = val;
}

template <int TOPK, int BLOCK_DIM>
__global__ void topk_kernel(int* input, int* output, int n) {
    __shared__ int shared_topk[BLOCK_DIM * TOPK];
    int local_topk[TOPK];

    for (int i = 0; i < TOPK; ++i) {
        local_topk[i] = INT32_MIN;
    }

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n;
         i += gridDim.x * blockDim.x) {
        update_topk<TOPK>(local_topk, input[i]);
    }

    for (int i = 0; i < TOPK; ++i) {
        shared_topk[TOPK * threadIdx.x + i] = local_topk[i];
    }
    __syncthreads();

    for (int stride = (BLOCK_DIM >> 1); stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            for (int i = 0; i < TOPK; ++i) {
                update_topk<TOPK>(local_topk, shared_topk[TOPK * (threadIdx.x + stride) + i]);
            }
        }
        __syncthreads();

        if (threadIdx.x < stride) {
            for (int i = 0; i < TOPK; ++i) {
                shared_topk[TOPK * threadIdx.x + i] = local_topk[i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        for (int i = 0; i < TOPK; ++i) {
            output[blockIdx.x * TOPK + i] = local_topk[i];
        }
    }
}

template <int TOPK>
void launch_topk(int* input, int* temp, int* output, int n, cudaStream_t stream) {
    constexpr int block_num = 64;
    constexpr int thread_num = 512;
    topk_kernel<TOPK, thread_num><<<block_num, thread_num, 0, stream>>>(input, temp, n);
    topk_kernel<TOPK, block_num><<<1, block_num, 0, stream>>>(temp, output, block_num * TOPK);
}
} // namespace Topk_One_Block_Reduce

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
    std::partial_sort_copy(input, input + n, output, output + k, std::greater<int>());
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

    double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
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
              << "\n";

    for (const auto& r : reports) {
        const double speedup = (r.avg_ms > 0.0) ? (baseline_ms / r.avg_ms) : 0.0;
        std::cout << std::left
                  << std::setw(28) << r.name
                  << std::setw(14) << r.avg_ms
                  << std::setw(18) << speedup
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

} // namespace bench_utils

int main() {
    constexpr int N = 10000000;
    constexpr int TOPK = 20;
    constexpr int warmup__iters = 10;
    constexpr int bench__iters = 20;
    constexpr int gpu_stage1_blocks = 512;

    std::vector<int> h_input(N);

    // 保留你当前文件中的初始化方式
    std::mt19937 rng(123);
    std::uniform_int_distribution<int> dist(0, INT_MAX);
    for (int i = 0; i < N; ++i) {
        h_input[i] = dist(rng);
    }

    std::vector<int> h_ref(TOPK);
    std::vector<int> h_out_baseline(TOPK);
    std::vector<int> h_out_quick(TOPK);
    std::vector<int> h_out_pq(TOPK);
    std::vector<int> h_out_gpu(TOPK);
    std::vector<int> h_temp_quick(N);

    int* d_input = nullptr;
    int* d_temp = nullptr;
    int* d_output = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_temp, static_cast<size_t>(gpu_stage1_blocks) * TOPK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, static_cast<size_t>(TOPK) * sizeof(int)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    CHECK_CUDA(cudaMemcpy(d_input,
                          h_input.data(),
                          static_cast<size_t>(N) * sizeof(int),
                          cudaMemcpyHostToDevice));

    bench_utils::cpu_reference_topk(h_input.data(), h_ref.data(), N, TOPK);

    std::vector<bench_utils::Report> reports;

    const double baseline_ms = bench_utils::benchmark_cpu_ms(
        [&]() {
            bench_utils::cpu_reference_topk(h_input.data(), h_out_baseline.data(), N, TOPK);
        },
        warmup__iters,
        bench__iters);

    auto baseline_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_baseline.data(), TOPK);
    reports.push_back({"baseline_cpu",
                       baseline_ms,
                       baseline_cmp.pass,
                       baseline_cmp.first_bad_idx,
                       baseline_cmp.ref_val,
                       baseline_cmp.test_val});

    const double quick_ms = bench_utils::benchmark_cpu_ms(
        [&]() {
            Topk_Quick_Select::topk_quick_select(
                h_input.data(), h_temp_quick.data(), h_out_quick.data(), N, TOPK);
        },
        warmup__iters,
        bench__iters);

    auto quick_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_quick.data(), TOPK);
    reports.push_back({"Topk_Quick_Select",
                       quick_ms,
                       quick_cmp.pass,
                       quick_cmp.first_bad_idx,
                       quick_cmp.ref_val,
                       quick_cmp.test_val});

    const double pq_ms = bench_utils::benchmark_cpu_ms(
        [&]() {
            Topk_Priority_Queue::topk_priority_queue(
                h_input.data(), h_out_pq.data(), N, TOPK);
        },
        warmup__iters,
        bench__iters);

    auto pq_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_pq.data(), TOPK);
    reports.push_back({"Topk_Priority_Queue",
                       pq_ms,
                       pq_cmp.pass,
                       pq_cmp.first_bad_idx,
                       pq_cmp.ref_val,
                       pq_cmp.test_val});

    const float gpu_ms = bench_utils::benchmark_gpu_ms(
        [&]() {
            // launch_topk 每次都会完整覆盖 d_temp 和 d_output
        },
        [&]() {
            Topk_One_Block_Reduce::launch_topk<TOPK>(d_input, d_temp, d_output, N, stream);
        },
        warmup__iters,
        bench__iters,
        stream);

    CHECK_CUDA(cudaMemcpy(h_out_gpu.data(),
                          d_output,
                          static_cast<size_t>(TOPK) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto gpu_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_gpu.data(), TOPK);
    reports.push_back({"Topk_One_Block_Reduce",
                       static_cast<double>(gpu_ms),
                       gpu_cmp.pass,
                       gpu_cmp.first_bad_idx,
                       gpu_cmp.ref_val,
                       gpu_cmp.test_val});

    std::cout << "N = " << N << ", TOPK = " << TOPK
              << ", warmup = " << warmup__iters
              << ", bench = " << bench__iters << "\n";
    std::cout << "(GPU time is kernel-only; H2D / D2H are excluded)\n\n";
    bench_utils::print_report_table(reports, baseline_ms);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
