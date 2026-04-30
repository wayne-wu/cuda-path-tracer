#include "GpuFramebuffer.h"

#include "CudaCheck.h"

namespace {
OptixImage2D makeFloat4Image(const float4* data, size_t width, size_t height) {
    OptixImage2D image = {};
    image.data = reinterpret_cast<CUdeviceptr>(data);
    image.width = static_cast<unsigned int>(width);
    image.height = static_cast<unsigned int>(height);
    image.pixelStrideInBytes = sizeof(float4);
    image.rowStrideInBytes = static_cast<unsigned int>(width * image.pixelStrideInBytes);
    image.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return image;
}

__global__ void framebufferToPboKernel(uchar4* pbo, const float4* color, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const int index = x + y * width;
    const float4 pixel = color[index];
    pbo[index] = make_uchar4(
        static_cast<unsigned char>(glm::clamp(pixel.x, 0.0f, 1.0f) * 255.0f),
        static_cast<unsigned char>(glm::clamp(pixel.y, 0.0f, 1.0f) * 255.0f),
        static_cast<unsigned char>(glm::clamp(pixel.z, 0.0f, 1.0f) * 255.0f),
        0);
}
}

void gpu::Framebuffer::resize(size_t width, size_t height) {
    this->width = width;
    this->height = height;

    const size_t pixelCount = width * height;
    accumulation.allocate(pixelCount);
    color.allocate(pixelCount);
    denoised.allocate(pixelCount);
    gbuffer.normal.allocate(pixelCount);
    gbuffer.albedo.allocate(pixelCount);

    clear();
}

void gpu::Framebuffer::clear() {
    const size_t pixelCount = width * height;
    if (pixelCount == 0) {
        return;
    }

    CUDA_CHECK(cudaMemset(accumulation.data(), 0, pixelCount * sizeof(float4)));
    CUDA_CHECK(cudaMemset(color.data(), 0, pixelCount * sizeof(float4)));
    CUDA_CHECK(cudaMemset(denoised.data(), 0, pixelCount * sizeof(float4)));
    CUDA_CHECK(cudaMemset(gbuffer.normal.data(), 0, pixelCount * sizeof(float4)));
    CUDA_CHECK(cudaMemset(gbuffer.albedo.data(), 0, pixelCount * sizeof(float4)));
}

void gpu::Framebuffer::reset() {
    width = 0;
    height = 0;
    accumulation.reset();
    color.reset();
    denoised.reset();
    gbuffer.normal.reset();
    gbuffer.albedo.reset();
}

OptixImage2D gpu::Framebuffer::colorInputImage() const {
    return makeFloat4Image(color.data(), width, height);
}

OptixImage2D gpu::Framebuffer::denoisedImage() const {
    return makeFloat4Image(denoised.data(), width, height);
}

OptixImage2D gpu::Framebuffer::albedoImage() const {
    return makeFloat4Image(gbuffer.albedo.data(), width, height);
}

OptixImage2D gpu::Framebuffer::normalImage() const {
    return makeFloat4Image(gbuffer.normal.data(), width, height);
}

void gpu::Framebuffer::copyToHost(std::vector<glm::vec3>& out) const {
    copyToHost(out, false);
}

void gpu::Framebuffer::copyToHost(std::vector<glm::vec3>& out, bool useDenoised) const {
    const size_t pixelCount = width * height;
    std::vector<float4> hostPixels(pixelCount);
    out.resize(pixelCount);

    if (pixelCount == 0) {
        return;
    }

    const float4* source = useDenoised ? denoised.data() : color.data();
    CUDA_CHECK(cudaMemcpy(hostPixels.data(), source, pixelCount * sizeof(float4), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < pixelCount; ++i) {
        out[i] = glm::vec3(hostPixels[i].x, hostPixels[i].y, hostPixels[i].z);
    }
}

void gpu::Framebuffer::copyToPbo(uchar4* pbo) const {
    if (!pbo || width == 0 || height == 0) {
        return;
    }

    const dim3 blockSize(8, 8);
    const dim3 gridSize(
        static_cast<unsigned int>((width + blockSize.x - 1) / blockSize.x),
        static_cast<unsigned int>((height + blockSize.y - 1) / blockSize.y));
    framebufferToPboKernel<<<gridSize, blockSize>>>(
        pbo,
        color.data(),
        static_cast<int>(width),
        static_cast<int>(height));
}
