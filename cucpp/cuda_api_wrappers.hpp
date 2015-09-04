#pragma once

#include "cuda_runtime.h"

#include "utils.h"

namespace cucpp {

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        throw cuda_error(status);
    }
}

template <typename T>
T * cuda_malloc(size_t n) {
    T * result;
    check(cudaMalloc((void**) &result, n * sizeof(T));
    return result;
}

}
