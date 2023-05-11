#include <mlas.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
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
            return  1;
        }
        void TrySimpleParallelFor(
            const std::ptrdiff_t total,
            const std::function<void(std::ptrdiff_t)>& fn) {
                for (size_t i = 0; i < total; i++) 
                {
                    fn(i);
                }
        };
    };
}; // namespace ov
// namespace cpu
int main() {
    size_t threadsTotal = ov::cpu::getTotalThreads();
    int M = 128;
    int N = 51864;
    int K = 384;
    bool trans_b = false;
    std::vector<float> originalB(N*K, 1);
    ov::cpu::ThreadPool fakePool;
    std::iota (originalB.begin(), originalB.end(), 0);
    int packed_b_size = MlasGemmPackBSize(N, K);
    std::vector<float> packedB(packed_b_size / sizeof(float), 0.0);
    std::cout << "B Packed Bytes " << packedB.size() << std::endl;
    MlasGemmPackB(CblasNoTrans,
                N,
                K,
                originalB.data(),
                trans_b ? K : N,
                static_cast<float*>(packedB.data()));
}