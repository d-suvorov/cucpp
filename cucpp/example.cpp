#include "cuda_runtime.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "device_vector.hpp"
#include "cublas_helper.hpp"

int main() {
    const size_t n = 1000;

    std::vector<double> h_a(n), h_b(n);
    std::iota(h_a.begin(),  h_a.end(),  0);
    std::iota(h_b.rbegin(), h_b.rend(), 0);

    cucpp::device_vector<double> d_a(h_a.data(), n);
    cucpp::device_vector<double> d_b(h_b.data(), n);

    cucpp::cublas_handle handle;
    // a = a + b
    cublas_axpy(handle, n, 1.0, d_a, 1, d_b, 1);

    cudaMemcpy(h_a.data(), d_a.get_data(), n * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (double i : h_a) {
        if (i != n - 1)
            std::cout << "Failed\n";
    }
    std::cout << "Passed\n";

    cudaDeviceReset();

    return 0;
}
