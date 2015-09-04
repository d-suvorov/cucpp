#pragma once

namespace cucpp {

template <typename T>
class device_vector {
    size_t length;
    T * data;

public:
    device_vector(size_t length);
    device_vector(device_vector const& rhs);
    device_vector(double * d_ptr, size_t length);
    device_vector(double * h_ptr, size_t length);
    ~device_vector();

    T * get_data();
    size_t get_length();
};

// TODO eliminate copy-paste

template<typename T>
device_vector<T>::device_vector(size_t length) : length(length) {
    cudaMalloc((void**) &data, length * sizeof(T));
}

template <typename T>
device_vector<T>::device_vector(device_vector const& rhs) : length(rhs.length) {
    cudaMalloc((void**) &data, length * sizeof(T));
    cudaMemcpy(data, rhs.get_data(), length * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
device_vector<T>::device_vector(double * d_ptr, size_t length) : length(length) {
    cudaMalloc((void**) &data, length * sizeof(T));
    cudaMemcpy(data, d_ptr, length * sizeof(T), cudaMemcpyDeviceToDevice);
}

template <typename T>
device_vector<T>::device_vector(double * h_ptr, size_t length) : length(length) {
    cudaMalloc((void**) &data, length * sizeof(T));
    cudaMemcpy(data, d_ptr, length * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
device_vector<T>::~device_vector() {
    cudaFree(data);
}

template <typename T>
T * device_vector<T>::get_data() {
    return data();
}

template <typename T>
size_t get_length() {
    return length;
}

}
