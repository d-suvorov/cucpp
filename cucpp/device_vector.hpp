#pragma once

#include "cuda_runtime.h"

namespace cucpp {

template <typename T>
class device_vector {
    size_t length;
    T * data;

public:
    device_vector(size_t length);
    device_vector(device_vector const & rhs);
    device_vector(T * h_ptr, size_t length);
    ~device_vector();

    T const * get_data() const;
    T       * get_data();

    size_t get_length() const;
};

// TODO eliminate copy-paste

template <typename T>
device_vector<T>::device_vector(size_t length) : length(length) {
    cudaMalloc((void**) &data, length * sizeof(T));
}

template <typename T>
device_vector<T>::device_vector(device_vector<T> const & rhs) : length(rhs.length) {
    size_t size = length * sizeof(T);
    cudaMalloc((void**) &data, size);
    cudaMemcpy(data, rhs.get_data(), size, cudaMemcpyDeviceToDevice);
}

template <typename T>
device_vector<T>::device_vector(T * h_ptr, size_t length) : length(length) {
    size_t size = length * sizeof(T);
    cudaMalloc((void**) &data, size);
    cudaMemcpy(data, h_ptr, size, cudaMemcpyHostToDevice);
}

template <typename T>
device_vector<T>::~device_vector() {
    cudaFree(data);
}

template <typename T>
T const * device_vector<T>::get_data() const {
    return data;
}

template <typename T>
T * device_vector<T>::get_data() {
    return data;
}

template <typename T>
size_t get_length() {
    return length;
}

}
