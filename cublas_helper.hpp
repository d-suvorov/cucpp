#pragma once

#include <complex>

#include "device_vector.hpp"
#include "device_matrix.hpp"

#include "boost/variant.hpp"

#include "cublas_v2.h"

namespace cucpp {

class cublas_handle {
    cublasHandle_t handle;

public:
    cublasHandle_t get() {
        return handle;
    }

    void set_pointer_mode() {
        cublasSetPointerMode(handle, mode);
    }

    cublas_handle(cublasPointerMode_t mode = CUBLAS_POINTER_MODE_HOST) {
        cublasStatus_t status;
        status = cublasCreate(&handle);
        status = cublasSetPointerMode(handle, mode);
    }

    ~cublas_handle() {
        cublasDestroy(handle);
    }

private:

}

template <typename T>
using scalar = boost::variant<T, T*>;

template <typename T>
T * get_pointer(scalar<T> a) {
    // TODO visitor?
    if (T * p_val = boost::get<T>(&a)) {
        return p_val;
    } else if (T ** p_ptr = boost::get<T*>(&a)) {
        return *p_ptr;
    }
    return nullptr; // never happens
}

// AXPY
//
void cublas_axpy(cublas_handle handle, size_t n, scalar<float> alpha,
                 device_vector<float> x, int incx, device_vector<float> y, int incy) {
    // TODO check if we can cast n to int
    cublasSaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.data(), incx, y.data(), incy);
}

void cublas_axpy(cublas_handle handle, size_t n, scalar<float> alpha,
                 device_vector<float> x, int incx, device_vector<float> y, int incy) {
    // TODO check if we can cast n to int
    cublasSaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.data(), incx, y.data(), incy);
}

/*
 TODO unimplemented
void cublas_axpy(cublas_handle handle, size_t n, scalar<cuComplex> alpha,
                 device_vector<cuComplex> x, int incx, device_vector<cuComplex> y, int incy) {
    // TODO check if we can cast n to int
    cublasSaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.data(), incx, y.data(), incy);
}

void cublas_axpy(cublas_handle handle, size_t n, scalar<cuDoubleComplex> alpha,
                 device_vector<cuDoubleComplex> x, int incx, device_vector<cuDoubleComplex> y, int incy) {
    // TODO check if we can cast n to int
    cublasSaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.data(), incx, y.data(), incy);
}
*/

// AXPY


}
