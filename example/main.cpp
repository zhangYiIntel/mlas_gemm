#include <mlas.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
int main() {
    int N = 384;
    int K = 384;
    bool trans_b = false;
    std::vector<float> originalB(384*384, 0);
    std::iota (originalB.begin(), originalB.end(), 0);
    int packed_b_size = MlasGemmPackBSize(N, K);
    std::vector<float> packedB(packed_b_size, 0);
    MlasGemmPackB(CblasNoTrans,
                N,
                K,
                originalB.data(),
                trans_b ? K : N,
                packedB.data());
    for (size_t i = 0; i < 256; i++) {
        if (i % 16 == 0)
            std::cout << std::endl; 
        std::cout << packedB[i] << "|";
    }
}