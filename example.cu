#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

#include "device_vector.hpp"

// c = a + b
__global__ void add(double * a, double * b, double *c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const size_t n = 1000;
    
    std::vector<double> h_a(n), h_b(n), h_c(n);
    std::iota(h_a.begin(),  h_a.end(),  0);
    std::iota(h_b.rbegin(), h_b.rend(), 0);

    cucpp::vector<double> d_a(n);
    cucpp::vector<double> d_b(n);
    cucpp::vector<double> d_c(n);

    cudaMemcpy(d_a.data, h_a.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b.data, h_b.data(), n * sizeof(double), cudaMemcpyHostToDevice);

    const unsigned block_dim = 256;
    const unsigned grid_dim = (n + block_dim - 1) / block_dim;
    add<<<grid_dim, block_dim>>>(d_a.data, d_b.data, d_c.data, n);

    cudaMemcpy(h_c.data(), d_c.data, n * sizeof(double), cudaMemcpyDeviceToHost);
    
    for (double i : h_c) {
        if (i != n - 1)
            std::cout << "Failed\n";
    }
    std::cout << "Passed\n";

    cudaDeviceReset();

    return 0;
}
