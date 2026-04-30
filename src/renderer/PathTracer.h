#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "glm/glm.hpp"

#include "scene/RenderScene.h"

constexpr int GBUFFER_TIME = 0;
constexpr int GBUFFER_NORMAL = 1;
constexpr int GBUFFER_POSITION = 2;

class PathTracer {
public:
    PathTracer();
    ~PathTracer();

    PathTracer(const PathTracer&) = delete;
    PathTracer& operator=(const PathTracer&) = delete;
    PathTracer(PathTracer&&) noexcept;
    PathTracer& operator=(PathTracer&&) noexcept;

    void initialize();
    void setScene(const RenderScene& scene);
    void setCamera(const RenderCamera& camera);
    void setTraceDepth(int depth);
    void resize(int width, int height);
    void reset();
    void renderSample();
    void renderSample(bool denoise);
    void copyImageToPbo(uchar4* pbo, bool denoise);
    void copyGBufferToPbo(uchar4* pbo, int mode);
    void copyImageToHost(std::vector<glm::vec3>& out);
    void copyImageToHost(std::vector<glm::vec3>& out, bool useDenoised);
    bool hasDenoisedImage() const;
    void copyImageToPbo(uchar4* pbo);
    void shutdown();

private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
