#pragma once

namespace cucpp {

template <typename T>
class device_vector {
    size_t length;
    T * data;

public:
    device_vector(size_t length);
    ~device_vector();

    T * get_data();
    size_t get_length();
};

template<typename T>
device_vector<T>::device_vector(size_t length) : length(length) {
    cudaMalloc((void**) &data, length * sizeof(T));
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
