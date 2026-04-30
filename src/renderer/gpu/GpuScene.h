#pragma once

#include <vector>

#include "scene/RenderScene.h"
#include "CudaBuffer.h"
#include "scene/GpuSceneTypes.h"
#include "CudaTexture2D.h"

namespace gpu {

class Scene {
public:
    void upload(const RenderScene& scene);
    void reset();

    SceneView view();

private:
    void uploadTextures(const RenderScene& scene);

    CudaBuffer<Mesh> meshes;
    CudaBuffer<Instance> instances;
    CudaBuffer<Primitive> primitives;
    CudaBuffer<Triangle> triangles;
    CudaBuffer<Material> materials;

    CudaBuffer<Vec3> vertices;
    CudaBuffer<Vec3> normals;
    CudaBuffer<uint32_t> indices;
    CudaBuffer<Vec2> uvs;
    CudaBuffer<Vec4> tangents;
    CudaBuffer<Bin> bins;
    CudaBuffer<int> binFaces;

    std::vector<CudaTexture2D> textures;
    CudaBuffer<cudaTextureObject_t> textureObjects;
    Environment environment;
};

} // namespace gpu

using GpuScene = gpu::Scene;
