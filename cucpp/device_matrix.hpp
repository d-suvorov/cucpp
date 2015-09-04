#pragma once

namespace cucpp {

template <typename T>
class device_matrix {
    size_t width;
    size_t height;
    size_t pitch;
    T * elements;

public:
    device_matrix(double * h_ptr, size_t width, size_t height, size_t pitch);
    ~device_matrix();
};

template <typename T>
device_matrix<T>::device_matrix(double * h_ptr, size_t width, size_t height, size_t pitch)
    : width(width)
    , height(height)
    , pitch(pitch)
{
    cudaMallocPitch();
    cudaMemcpy2D();
}

}
