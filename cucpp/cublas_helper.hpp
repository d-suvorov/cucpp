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

    void set_pointer_mode(cublasPointerMode_t mode) {
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

};

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
class scalar_result {
    T * ptr;

public:
    scalar_result() {
        cudaMalloc((void**) &ptr, sizeof(T));
    }

    T * get_ptr() {
        return ptr;
    }

    T get(cudaStream_t stream = 0) {
        T result;
        cudaMemcpyAsync(&result, ptr, sizeof(T), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        return result;
    }

    ~scalar_result() {
        cudaFree(ptr);
    }
};

// TODO get rid of copy-paste vvvvvvvv

// AMAX
size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<float> & x, int incx) {
    int result;
    cublasIsamax(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<double> & x, int incx) {
    int result;
    cublasIdamax(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<cuComplex> & x, int incx) {
    int result;
    cublasIcamax(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> & x, int incx) {
    int result;
    cublasIzamax(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}
// AMAX

// AMIN
size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<float> & x, int incx) {
    int result;
    cublasIsamin(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<double> & x, int incx) {
    int result;
    cublasIdamin(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<cuComplex> & x, int incx) {
    int result;
    cublasIcamin(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> & x, int incx) {
    int result;
    cublasIzamin(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}
// AMIN

// ASUM
float cublas_asum(cublas_handle & handle, size_t n, device_vector<float> & x, int incx) {
    float result;
    cublasSasum(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return result;
}

double cublas_asum(cublas_handle & handle, size_t n, device_vector<double> & x, int incx) {
    double result;
    cublasDasum(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return result;
}

float cublas_asum(cublas_handle & handle, size_t n, device_vector<cuComplex> & x, int incx) {
    float result;
    cublasScasum(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return result;
}

double cublas_asum(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> & x, int incx) {
    double result;
    cublasDzasum(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return result;
}
// ASUM

// AXPY
void cublas_axpy(cublas_handle & handle, size_t n, scalar<float> alpha,
                 device_vector<float> & x, int incx, device_vector<float> & y, int incy) {
    cublasSaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}

void cublas_axpy(cublas_handle & handle, size_t n, scalar<double> alpha,
                 device_vector<double> & x, int incx, device_vector<double> & y, int incy) {
    cublasDaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}

void cublas_axpy(cublas_handle handle, size_t n, scalar<cuComplex> alpha,
                 device_vector<cuComplex> x, int incx, device_vector<cuComplex> y, int incy) {
    cublasCaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}

void cublas_axpy(cublas_handle handle, size_t n, scalar<cuDoubleComplex> alpha,
                 device_vector<cuDoubleComplex> x, int incx, device_vector<cuDoubleComplex> y, int incy) {
    cublasZaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}
// AXPY


}
