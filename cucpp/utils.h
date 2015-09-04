#pragma once

#include "cuda_runtime.h"

#include <exception>

namespace cucpp {

class cuda_error : public std::exception {
    cudaError_t code;

    cuda_error(cudaError_t code) : code(code) {}

public:
    cudaError_t get() {
        return code;
    }

    virtual const char* what() const throw() {
        return cudaGetErrorString(code);
    }
};

}
