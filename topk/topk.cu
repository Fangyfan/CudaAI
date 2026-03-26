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

namespace Topk_Block_Reduce {
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
        local_topk[i] = INT_MIN;
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
    constexpr int block_num = 128;
    constexpr int thread_num = 32;
    topk_kernel<TOPK, thread_num><<<block_num, thread_num, 0, stream>>>(input, temp, n);
    topk_kernel<TOPK, block_num><<<1, block_num, 0, stream>>>(temp, output, block_num * TOPK);
}
} // namespace Topk_Block_Reduce

namespace Topk_Block_Reduce_Optimized {
template <int TOPK>
__device__ __forceinline__ void init_topk(int* topk) {
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
        topk[i] = INT_MIN;
    }
}

// topk 按降序排列：topk[0] >= topk[1] >= ... >= topk[TOPK-1]
template <int TOPK>
__device__ __forceinline__ void update_topk(int* topk, int val) {
    if (val <= topk[TOPK - 1]) {
        return;
    }

#pragma unroll
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

template <int TOPK>
__device__ __forceinline__ void merge_topk(int* dst, const int* src) {
#pragma unroll
    for (int i = 0; i < TOPK; ++i) {
        update_topk<TOPK>(dst, src[i]);
    }
}

// warp 内规约：每个 lane 都维护自己的 topk，经过 butterfly 之后，
// warp 内所有 lane 都会得到相同的 warp topk。
// 这样做的好处是：
// 1. 不需要在线程级别把所有 topk 都写进 shared memory
// 2. 不需要 block 级树规约那么多 __syncthreads()
template <int TOPK, int WARP_SIZE = 32>
__device__ __forceinline__ void warp_reduce_topk(int* local_topk) {
    int peer_topk[TOPK];

#pragma unroll
    for (int delta = (WARP_SIZE >> 1); delta > 0; delta >>= 1) {
#pragma unroll
        for (int i = 0; i < TOPK; ++i) {
            peer_topk[i] = __shfl_xor_sync(0xffffffff, local_topk[i], delta, WARP_SIZE);
        }
        merge_topk<TOPK>(local_topk, peer_topk);
    }
}

template <int TOPK, int BLOCK_DIM>
__global__ void topk_kernel(const int* __restrict__ input, int* __restrict__ output, int n) {
    static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM must be multiple of 32");

    constexpr int WARPS_PER_BLOCK = BLOCK_DIM / 32;

     // 每个 warp 只存一份 warp topk
    __shared__ int shared_warp_topk[WARPS_PER_BLOCK][TOPK];

    int local_topk[TOPK];
    init_topk<TOPK>(local_topk);

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    const int total_threads = gridDim.x * blockDim.x;

    // 全局内存读取：int4 向量化
    const int n4 = n >> 2;
    const int4* input4 = reinterpret_cast<const int4*>(input);
    for (int i = global_thread_id; i < n4; i += total_threads) {
        int4 v = input4[i];
        update_topk<TOPK>(local_topk, v.x);
        update_topk<TOPK>(local_topk, v.y);
        update_topk<TOPK>(local_topk, v.z);
        update_topk<TOPK>(local_topk, v.w);
    }
    for (int i = (n4 << 2) + global_thread_id; i < n; i += total_threads) {
        update_topk<TOPK>(local_topk, input[i]);
    }

    // warp 内归约 -> warp topk
    warp_reduce_topk<TOPK, 32>(local_topk);

    // 每个 warp 只让 lane0 写出一份 warp topk
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < TOPK; ++i) {
            shared_warp_topk[warp][i] = local_topk[i];
        }
    }
    __syncthreads();

    // warp0 归并所有 warp topk -> block topk
    if (warp == 0) {
        int block_topk[TOPK];
        init_topk<TOPK>(block_topk);

        // 只有前 WARPS_PER_BLOCK 个 lane 负责加载 warp 结果
        if (lane < WARPS_PER_BLOCK) {
#pragma unroll
            for (int i = 0; i < TOPK; ++i) {
                block_topk[i] = shared_warp_topk[lane][i];
            }
        }

        // 其他 lane 用 -INF 填充，不影响最终结果
        if (lane >= WARPS_PER_BLOCK) {
            init_topk<TOPK>(block_topk);
        }

        // warp0 内再做一次 topk 归约
        warp_reduce_topk<TOPK, WARPS_PER_BLOCK>(block_topk);

        if (lane == 0) {
#pragma unroll
            for (int i = 0; i < TOPK; ++i) {
                output[blockIdx.x * TOPK + i] = block_topk[i];
            }
        }
    }
}

template <int TOPK>
void launch_topk(int* input, int* temp, int* output, int n, cudaStream_t stream) {
    constexpr int block_num = 128;
    constexpr int thread_num = 32;
    topk_kernel<TOPK, thread_num><<<block_num, thread_num, 0, stream>>>(input, temp, n);
    topk_kernel<TOPK, block_num><<<1, block_num, 0, stream>>>(temp, output, block_num * TOPK);
}
} // namespace Topk_Block_Reduce_Optimized

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
              << std::setw(34) << "name"
              << std::setw(14) << "avg_ms"
              << std::setw(18) << "speedup_vs_base"
              << std::setw(10) << "correct"
              << "\n";

    for (const auto& r : reports) {
        const double speedup = (r.avg_ms > 0.0) ? (baseline_ms / r.avg_ms) : 0.0;
        std::cout << std::left
                  << std::setw(34) << r.name
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
    constexpr int gpu_stage1_blocks = 64;

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
    std::vector<int> h_out_gpu_opt(TOPK);
    std::vector<int> h_temp_quick(N);

    int* d_input = nullptr;
    int* d_temp = nullptr;
    int* d_output = nullptr;
    int* d_temp_opt = nullptr;
    int* d_output_opt = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, static_cast<size_t>(N) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_temp, static_cast<size_t>(gpu_stage1_blocks) * TOPK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output, static_cast<size_t>(TOPK) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_temp_opt, static_cast<size_t>(gpu_stage1_blocks) * TOPK * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_output_opt, static_cast<size_t>(TOPK) * sizeof(int)));

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
    reports.push_back({"baseline_cpu_partial_sort_copy",
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
            Topk_Block_Reduce::launch_topk<TOPK>(d_input, d_temp, d_output, N, stream);
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
    reports.push_back({"Topk_Block_Reduce",
                       static_cast<double>(gpu_ms),
                       gpu_cmp.pass,
                       gpu_cmp.first_bad_idx,
                       gpu_cmp.ref_val,
                       gpu_cmp.test_val});

    const float gpu_opt_ms = bench_utils::benchmark_gpu_ms(
        [&]() {
            // launch_topk 每次都会完整覆盖 d_temp_opt 和 d_output_opt
        },
        [&]() {
            Topk_Block_Reduce_Optimized::launch_topk<TOPK>(d_input, d_temp_opt, d_output_opt, N, stream);
        },
        warmup__iters,
        bench__iters,
        stream);

    CHECK_CUDA(cudaMemcpy(h_out_gpu_opt.data(),
                          d_output_opt,
                          static_cast<size_t>(TOPK) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    auto gpu_opt_cmp = bench_utils::compare_topk_exact(h_ref.data(), h_out_gpu_opt.data(), TOPK);
    reports.push_back({"Topk_Block_Reduce_Optimized",
                       static_cast<double>(gpu_opt_ms),
                       gpu_opt_cmp.pass,
                       gpu_opt_cmp.first_bad_idx,
                       gpu_opt_cmp.ref_val,
                       gpu_opt_cmp.test_val});

    std::cout << "N = " << N << ", TOPK = " << TOPK
              << ", warmup = " << warmup__iters
              << ", bench = " << bench__iters << "\n";
    std::cout << "(GPU time is kernel-only; H2D / D2H are excluded)\n\n";
    bench_utils::print_report_table(reports, baseline_ms);

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_temp_opt));
    CHECK_CUDA(cudaFree(d_output_opt));
    CHECK_CUDA(cudaStreamDestroy(stream));
    return 0;
}
