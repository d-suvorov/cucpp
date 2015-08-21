#pragma once

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
    if (T * p_val = boost::get<T>(&a)) {
        return p_val;
    } else if (T ** p_ptr = boost::get<T*>(&a)) {
        return *p_ptr;
    }
    return nullptr; // never happens
}

template <typename T>
void cublas_axpy(cublas_handle handle, scalar<T> alpha, device_vector<T> x, inc incx, device_vector<T> y, int incy) {
    
}


}
