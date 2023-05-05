#include <mlas.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include "ie_parallel.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
using namespace ngraph;
namespace ov
{
    namespace cpu {
        class ThreadPool {
            public:
                ThreadPool(){
                    std::cout << "Fake ThreadTool" << std::endl;
                }
        };
        size_t getTotalThreads() {
            return  parallel_get_max_threads();
        }
        void TrySimpleParallelFor(
            const std::ptrdiff_t total,
            const std::function<void(std::ptrdiff_t)>& fn) {
                    // #pragma omp parallel 
                    // {
                    //     #pragma omp for
                    //     for (size_t i = 0; i < total; i++) 
                    //     {
                    //         fn(i);
                    //     }
                    // }
            parallel_for(total, fn);
        };
    };
}; // namespace ov
// namespace cpu
int main() {
    size_t threadsTotal = ov::cpu::getTotalThreads();
    // #pragma omp parallel for 
    //     for (size_t i = 0; i < 5000; i++) {
    //         int id = omp_get_thread_num();
    //         std::cout <<" omp id" << id << std::endl;
    //     }
    std::cout << "Total Threads " << threadsTotal << std::endl;
    int M = 128;
    int N = 51864;
    int K = 384;
    bool trans_b = false;
    std::vector<float> originalB(N*K, 1);
    ov::cpu::ThreadPool fakePool;
    std::iota (originalB.begin(), originalB.end(), 0);
    int packed_b_size = MlasGemmPackBSize(N, K);
    // std::vector<uint8_t> packedB(packed_b_size, 0);
    runtime::AlignedBuffer alignedPackB(packed_b_size, 64);
    std::cout << "B Packed Bytes " << alignedPackB.size() << std::endl;
    MlasGemmPackB(CblasNoTrans,
                N,
                K,
                originalB.data(),
                trans_b ? K : N,
                alignedPackB.get_ptr());
    // std::vector<float> A(M*K, 1);
    runtime::AlignedBuffer alignedA(M*K*4, 64);
    runtime::AlignedBuffer alignedC(M*N*4, 64);
    std::cout << "A Size " << M * K << std::endl;
    // std::vector<float> C(M*N, 0.5);
    std::cout << "C Size " << M * N << std::endl;
    std::vector<MLAS_SGEMM_DATA_PARAMS> data(1);
    size_t i = 0;
    data[i].BIsPacked = true;
    data[i].A = alignedA.get_ptr<float>();
    data[i].lda = K;
    data[i].B = alignedPackB.get_ptr<float>();
    data[i].ldb = N;
    data[i].C = alignedC.get_ptr<float>();
    data[i].ldc = N;
    data[i].alpha = 1;
    data[i].beta = 0.0f;
    auto begin = std::chrono::high_resolution_clock::now();
    size_t trials = 1000;
    std::cout << "A ptr " << data[i].A << std::endl;
    std::cout << "B ptr " << data[i].B << std::endl;
    std::cout << "C ptr " << data[i].C << std::endl;

    for (size_t i = 0; i < trials; i++) {
        MlasGemmBatch(CblasNoTrans, CblasNoTrans, M, N, K, data.data(), 1, &fakePool);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << total_time / trials << std::endl;
}