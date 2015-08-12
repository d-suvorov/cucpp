#pragma once

namespace cucpp {

template <typename T>
struct vector {
    T * data;

    vector(size_t n);

    ~vector();
};

template<typename T>
vector<T>::vector(size_t n) {
    cudaMalloc((void**) &data, n * sizeof(T));
}

template<typename T>
vector<T>::~vector() {
    cudaFree(data);
}

}
