#pragma once

#include "cuda_runtime.h"
#include "CudaCheck.h"

#include <cstring>
#include <utility>


class CudaTexture2D {
public:
    CudaTexture2D() = default;
    ~CudaTexture2D() { reset(); }

    CudaTexture2D(const CudaTexture2D&) = delete;
    CudaTexture2D& operator=(const CudaTexture2D&) = delete;

    CudaTexture2D(CudaTexture2D&& other) noexcept {
        moveFrom(other);
    }

    CudaTexture2D& operator=(CudaTexture2D&& other) noexcept {
        if (this != &other) {
            reset();
            moveFrom(other);
        }
        return *this;
    }

    void uploadRGBA8(int width, int height, const unsigned char* data) {
        reset();

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        CUDA_CHECK(cudaMallocArray(&cudaArray, &channelDesc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(
            cudaArray,
            0,
            0,
            data,
            width * sizeof(uchar4),
            width * sizeof(uchar4),
            height,
            cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeWrap;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeNormalizedFloat;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

        this->width = width;
        this->height = height;
        this->channels = 4;
    }

    void uploadRGBA32F(int width, int height, const float* data) {
        reset();

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
        CUDA_CHECK(cudaMallocArray(&cudaArray, &channelDesc, width, height));
        CUDA_CHECK(cudaMemcpy2DToArray(
            cudaArray,
            0,
            0,
            data,
            width * sizeof(float4),
            width * sizeof(float4),
            height,
            cudaMemcpyHostToDevice));

        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cudaArray;

        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeWrap;
        texDesc.addressMode[1] = cudaAddressModeClamp;
        texDesc.filterMode = cudaFilterModeLinear;
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 1;

        CUDA_CHECK(cudaCreateTextureObject(&textureObject, &resDesc, &texDesc, NULL));

        this->width = width;
        this->height = height;
        this->channels = 4;
    }

    void reset() noexcept {
        if (textureObject) {
            cudaDestroyTextureObject(textureObject);
            textureObject = 0;
        }
        if (cudaArray) {
            cudaFreeArray(cudaArray);
            cudaArray = nullptr;
        }
        width = 0;
        height = 0;
        channels = 0;
    }

    cudaTextureObject_t object() const { return textureObject; }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    int getChannels() const { return channels; }

private:
    void moveFrom(CudaTexture2D& other) noexcept {
        textureObject = std::exchange(other.textureObject, 0);
        cudaArray = std::exchange(other.cudaArray, nullptr);
        width = std::exchange(other.width, 0);
        height = std::exchange(other.height, 0);
        channels = std::exchange(other.channels, 0);
    }

    // pointer to the CUDA array
    cudaTextureObject_t textureObject = 0;
    // CUDA array on device
    cudaArray_t cudaArray = nullptr;
    int width = 0;
    int height = 0;
    int channels = 0;
};
