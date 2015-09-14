#include "cuda_runtime.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "device_vector.hpp"
#include "cublas_helper.hpp"

void test() {
    const size_t n = 1000;

    std::vector<double> h_a(n), h_b(n);
    std::iota(h_a.begin(),  h_a.end(),  0);
    std::iota(h_b.rbegin(), h_b.rend(), 0);

    cucpp::device_vector<double> d_a(h_a.data(), n);
    cucpp::device_vector<double> d_b(h_b.data(), n);

    cucpp::cublas_handle handle;
    double alpha = 1.0;
    // b[i] = alpha * a[i] + b[i] , forall i = 0 .. n - 1
    cublas_axpy(handle, n, alpha, d_a, 1, d_b, 1);

    cudaMemcpy(h_b.data(), d_b.get_data(), n * sizeof(double), cudaMemcpyDeviceToHost);

    bool ok = std::all_of(h_b.begin(), h_b.end(), [n](double x){ return x == n - 1; });
    std::cout << (ok ? "Passed" : "Failed") << std::endl;
}

void test1() {
    std::vector<double> h_a = {1, 1, 2, 1, 1};
    cucpp::device_vector<double> d_a(h_a.data(), h_a.size());
    cucpp::cublas_handle handle;
    size_t max_idx = cublas_iamax(handle, h_a.size(), d_a, 1);
    std::cout << (max_idx == 2 ? "Passed" : "Failed") << std::endl;
}

void test2() {
    std::vector<double> h_a = {1, 2, 3, 4, 5};
    cucpp::device_vector<double> d_a(h_a.data(), h_a.size());
    cucpp::cublas_handle handle;
    double sum = cublas_asum(handle, h_a.size(), d_a, 1);
    std::cout << (sum == 15 ? "Passed" : "Failed") << std::endl;
}

int main() {
    test();
    test1();
    test2();

    cudaDeviceReset();

    return 0;
}
