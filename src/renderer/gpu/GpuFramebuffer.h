#pragma once

#include <cuda_runtime.h>
#include <optix.h>

#include <cstddef>
#include <vector>

#include "glm/glm.hpp"

#include "CudaBuffer.h"

namespace gpu {

struct GBufferView {
    float4* normal;
    float4* albedo;
};

struct GBuffer {
    CudaBuffer<float4> normal;
    CudaBuffer<float4> albedo;

    GBufferView view() {
        GBufferView v;
        v.normal = normal.data();
        v.albedo = albedo.data();
        return v;
    }
};

class Framebuffer {
public:
    void resize(size_t width, size_t height);
    void clear();
    void reset();

    OptixImage2D colorInputImage() const;
    OptixImage2D denoisedImage() const;
    OptixImage2D albedoImage() const;
    OptixImage2D normalImage() const;
    void copyToHost(std::vector<glm::vec3>& out) const;
    void copyToHost(std::vector<glm::vec3>& out, bool useDenoised) const;
    void copyToPbo(uchar4* pbo) const;

    CudaBuffer<float4> accumulation;
    CudaBuffer<float4> color;
    CudaBuffer<float4> denoised;

    GBuffer gbuffer;

private:
    size_t width = 0;
    size_t height = 0;

};
}

using GpuFramebuffer = gpu::Framebuffer;
