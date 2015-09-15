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


template <typename CublasRoutine, typename T>
size_t cublas_imaxmin_impl(cublas_handle & handle, size_t n, device_vector<T> & x, int incx,
                           CublasRoutine routine) {
    int result;
    routine(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return static_cast<size_t>(result - 1);
}

// AMAX
size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<float> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIsamax);
}

size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<double> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIdamax);
}

size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<cuComplex> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIcamax);
}

size_t cublas_iamax(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIzamax);
}
// AMAX

// AMIN
size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<float> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIsamin);
}

size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<double> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIdamin);
}

size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<cuComplex> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIcamin);
}

size_t cublas_iamin(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> & x, int incx) {
    return cublas_imaxmin_impl(handle, n, x, incx, cublasIzamin);
}
// AMIN

// ASUM
template <typename CublasRoutine, typename T, typename Ret>
Ret cublas_asum_impl(cublas_handle & handle, size_t n, device_vector<T> & x, int incx,
                     CublasRoutine routine) {
    Ret result;
    routine(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return result;
}

float cublas_asum(cublas_handle & handle, size_t n, device_vector<float> & x, int incx) {
    return cublas_asum_impl<decltype(cublasSasum), float, float>(handle, n, x, incx, cublasSasum);
}

double cublas_asum(cublas_handle & handle, size_t n, device_vector<double> & x, int incx) {
    return cublas_asum_impl<decltype(cublasDasum), double, double>(handle, n, x, incx, cublasDasum);
}

float cublas_asum(cublas_handle & handle, size_t n, device_vector<cuComplex> & x, int incx) {
    return cublas_asum_impl<decltype(cublasScasum), cuComplex, float>(handle, n, x, incx, cublasScasum);
}

double cublas_asum(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> & x, int incx) {
    return cublas_asum_impl<decltype(cublasDzasum), cuDoubleComplex, double>(handle, n, x, incx, cublasDzasum);
}
// ASUM

// AXPY
void cublas_axpy(cublas_handle & handle, size_t n, scalar<float> alpha,
                 device_vector<float> const & x, int incx,
                 device_vector<float>       & y, int incy) {
    cublasSaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}

void cublas_axpy(cublas_handle & handle, size_t n, scalar<double> alpha,
                 device_vector<double> const & x, int incx,
                 device_vector<double>       & y, int incy) {
    cublasDaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}

void cublas_axpy(cublas_handle handle, size_t n, scalar<cuComplex> alpha,
                 device_vector<cuComplex> const & x, int incx,
                 device_vector<cuComplex>       & y, int incy) {
    cublasCaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}

void cublas_axpy(cublas_handle handle, size_t n, scalar<cuDoubleComplex> alpha,
                 device_vector<cuDoubleComplex> const & x, int incx,
                 device_vector<cuDoubleComplex>       & y, int incy) {
    cublasZaxpy(handle.get(), static_cast<int>(n), get_pointer(alpha), x.get_data(), incx, y.get_data(), incy);
}
// AXPY

// COPY
void cublas_copy(cublas_handle & handle, size_t n,
                 device_vector<float> const & x, int incx,
                 device_vector<float>       & y, int incy) {
    cublasScopy(handle.get(), static_cast<int>(n), x.get_data(), incx, y.get_data(), incy);
}

void cublas_copy(cublas_handle & handle, size_t n,
                 device_vector<double> const & x, int incx,
                 device_vector<double>       & y, int incy) {
    cublasDcopy(handle.get(), static_cast<int>(n), x.get_data(), incx, y.get_data(), incy);
}

void cublas_copy(cublas_handle & handle, size_t n,
                 device_vector<cuComplex> const & x, int incx,
                 device_vector<cuComplex>       & y, int incy) {
    cublasCcopy(handle.get(), static_cast<int>(n), x.get_data(), incx, y.get_data(), incy);
}

void cublas_copy(cublas_handle & handle, size_t n,
                 device_vector<cuDoubleComplex> const & x, int incx,
                 device_vector<cuDoubleComplex>       & y, int incy) {
    cublasZcopy(handle.get(), static_cast<int>(n), x.get_data(), incx, y.get_data(), incy);
}
// COPY

// DOT
template <typename CublasRoutine, typename T, typename Ret>
Ret cublas_dot_impl(cublas_handle & handle, size_t n,
                    device_vector<T> const & x, int incx,
                    device_vector<T> const & y, int incy, CublasRoutine routine) {
    Ret result;
    routine(handle.get(), static_cast<int>(n), x.get_data(), incx, y.get_data(), incy, &result);
    return result;
}

float cublas_dot(cublas_handle & handle, size_t n,
                 device_vector<float> const & x, int incx,
                 device_vector<float> const & y, int incy) {
    return cublas_dot_impl<decltype(cublasSdot), float, float>(handle, n, x, incx, y, incy, cublasSdot);
}

double cublas_dot(cublas_handle & handle, size_t n,
                  device_vector<double> const & x, int incx,
                  device_vector<double> const & y, int incy) {
    return cublas_dot_impl<decltype(cublasDdot), double, double>(handle, n, x, incx, y, incy, cublasDdot);
}

cuComplex cublas_dot(cublas_handle & handle, size_t n,
                     device_vector<cuComplex> const & x, int incx,
                     device_vector<cuComplex> const & y, int incy, bool conjugate = false) {
    auto routine = conjugate ? cublasCdotc : cublasCdotu;
    return cublas_dot_impl<decltype(routine), cuComplex, cuComplex>(handle, n, x, incx, y, incy, routine);
}

cuDoubleComplex cublas_dot(cublas_handle & handle, size_t n,
                           device_vector<cuDoubleComplex> const & x, int incx,
                           device_vector<cuDoubleComplex> const & y, int incy, bool conjugate = false) {
    auto routine = conjugate ? cublasZdotc : cublasZdotu;
    return cublas_dot_impl<decltype(routine), cuDoubleComplex, cuDoubleComplex>
        (handle, n, x, incx, y, incy, routine);
}
// DOT

// NRM2
template <typename CublasRoutine, typename T, typename Ret>
Ret cublas_nrm2_impl(cublas_handle & handle, size_t n,
                     device_vector<T> const & x, int incx,
                     CublasRoutine routine) {
    Ret result;
    routine(handle.get(), static_cast<int>(n), x.get_data(), incx, &result);
    return result;
}

float cublas_nrm2(cublas_handle & handle, size_t n, device_vector<float> const & x, int incx) {
    return cublas_nrm2_impl<decltype(cublasSnrm2), float, float>(handle, n, x, incx, cublasSnrm2);
}

double cublas_nrm2(cublas_handle & handle, size_t n, device_vector<double> const & x, int incx) {
    return cublas_nrm2_impl<decltype(cublasDnrm2), double, double>(handle, n, x, incx, cublasDnrm2);
}

float cublas_nrm2(cublas_handle & handle, size_t n, device_vector<cuComplex> const & x, int incx) {
    return cublas_nrm2_impl<decltype(cublasScnrm2), cuComplex, float>(handle, n, x, incx, cublasScnrm2);
}

double cublas_nrm2(cublas_handle & handle, size_t n, device_vector<cuDoubleComplex> const & x, int incx) {
    return cublas_nrm2_impl<decltype(cublasDznrm2), cuDoubleComplex, double>(handle, n, x, incx, cublasDznrm2);
}
// NRM2

}
