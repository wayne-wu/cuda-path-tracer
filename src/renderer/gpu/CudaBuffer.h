#pragma once

#include <cstddef>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

template <class T>
class CudaBuffer {
public:
    CudaBuffer() = default;
    ~CudaBuffer() = default;

    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    CudaBuffer(CudaBuffer&&) noexcept = default;
    CudaBuffer& operator=(CudaBuffer&&) noexcept = default;

    void allocate(size_t count) {
        data_.resize(count);
    }

    void upload(const std::vector<T>& source) {
        data_ = source;
    }

    void reset() {
        data_.clear();
        data_.shrink_to_fit();
    }

    T* data() { return data_.empty() ? nullptr : thrust::raw_pointer_cast(data_.data()); }
    const T* data() const { return data_.empty() ? nullptr : thrust::raw_pointer_cast(data_.data()); }
    size_t size() const { return data_.size(); }
    bool empty() const { return data_.empty(); }

private:
    thrust::device_vector<T> data_;
};
