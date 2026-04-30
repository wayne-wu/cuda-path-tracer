// scene/GpuSceneTypes.h
#pragma once

#include <cuda_runtime.h>
#include "glm/glm.hpp"

namespace gpu {

using Offset = int;
using Vec2 = glm::vec2;
using Vec3 = glm::vec3;
using Vec4 = glm::vec4;
using Mat4 = glm::mat4;
using Color = Vec3;

enum class GeometryType {
    Sphere,
    Cube,
    Mesh,
};

struct Ray {
    Vec3 origin;
    Vec3 direction;
    Vec3 inv_dir;
};

struct Camera {
    glm::ivec2 resolution = glm::ivec2(800, 800);
    Vec3 position = Vec3(0.0f);
    Vec3 lookAt = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 view = Vec3(0.0f, 0.0f, -1.0f);
    Vec3 up = Vec3(0.0f, 1.0f, 0.0f);
    Vec3 right = Vec3(1.0f, 0.0f, 0.0f);
    Vec2 fov = Vec2(45.0f);
    Vec2 pixelLength = Vec2(1.0f);
    Mat4 viewMat = Mat4(1.0f);
    Mat4 projMat = Mat4(1.0f);
};

struct PathSegment {
    Ray ray;
    Color color = Color(1.0f);
    int pixelIndex = -1;
    int remainingBounces = 0;
};

struct Hit {
    int geomId = -1;
    int meshId = -1;
    int primId = -1;
    int binId = -1;
    int faceId = -1;
    Vec3 bary = Vec3(0.0f);
    float t = -1.0f;       // world-space t
    float objectT = -1.0f; // object-space t
};

struct ShadeableIntersection {
    int materialId = -1;
    Vec3 surfaceNormal = Vec3(0.0f);
    Vec2 uv = Vec2(0.0f);
    Vec4 tangent = Vec4(1.0f, 0.0f, 0.0f, 1.0f);
    float t = -1.0f;  // world-space t
    int padding = 0;
};

struct GBufferPixel {
    Vec3 n = Vec3(0.0f);
    Vec3 albedo = Vec3(0.0f);
};

struct Mesh {
    int prim_count = 0;
    int prim_offset = 0;
};

struct Instance {
    GeometryType geometryType = GeometryType::Mesh;
    int meshId = -1;
    int materialId = -1;
    Mat4 transform = Mat4(1.0f);
    Mat4 inverseTransform = Mat4(1.0f);
    Mat4 inverseTranspose = Mat4(1.0f);
    Vec3 bboxMin = Vec3(0.0f);
    Vec3 bboxMax = Vec3(0.0f);
};

struct Triangle {
    Vec3 v0;
    Vec3 v1;
    Vec3 v2;
};

struct Primitive {
    int count = 0;
    int num_vertices = 0;
    Offset i_offset = 0;
    Offset v_offset = 0;
    Offset face_offset = -1;
    Offset bin_offset = -1;
    Offset bf_offset = -1;
    Offset n_offset = -1;
    Offset uv_offset = -1;
    Offset t_offset = -1;
    Vec3 bbox_max = Vec3(0.0f);
    Vec3 bbox_min = Vec3(0.0f);
    Mat4 pivot_xform = Mat4(1.0f);
};

struct Bin {
    Vec3 bbox_min;  // 3
    Vec3 bbox_max;  // 3
    int childIndex; // 1
    int startIndex; // 1
    int endIndex;   // 1
    int depth;      // 1
    Vec2 padding;   // 3
};

struct TextureRef {
    int index = -1;
    int texCoord = 0;
};

struct Environment {
    bool enabled = false;
    float intensity = 1.0f;
    float rotation = 0.0f;
    TextureRef texture;
};

struct Material {
    Vec3 baseColor = Vec3(1.0f);
    TextureRef baseColorTexture;

    float metallic = 0.0f;
    float roughness = 0.5f;
    TextureRef metallicRoughnessTexture;

    Vec3 emissiveColor = Vec3(0.0f);
    float emissiveStrength = 0.0f;
    TextureRef emissiveTexture;

    TextureRef normalTexture;
    float normalScale = 1.0f;

    float transmission = 0.0f;          // your hasRefractive equivalent
    float ior = 1.5f;

    float alpha = 1.0f;
    bool doubleSided = false;
    bool useRoughSpecular = false;
};

struct PrimitiveDataView {
    Primitive* primitives = nullptr;
    Triangle* triangles = nullptr;
    Vec3* vertices = nullptr;
    Vec3* normals = nullptr;
    uint32_t* indices = nullptr;
    Vec2* uvs = nullptr;
    Vec4* tangents = nullptr;
    Bin* bins = nullptr;
    int* binFaces = nullptr;
};

struct SceneView {
    Mesh* meshes = nullptr;
    Instance* instances = nullptr;
    Material* materials = nullptr;
    cudaTextureObject_t* textureObjects = nullptr;

    PrimitiveDataView primData;
    Environment environment;

    int meshCount = 0;
    int instanceCount = 0;
    int materialCount = 0;
    int textureCount = 0;
};

struct isPathCompleted {
    __host__ __device__
    bool operator()(const PathSegment& pathSegment) const {
        return pathSegment.remainingBounces <= 0;
    }
};

struct compareIntersections {
    __host__ __device__
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) const {
        return a.materialId > b.materialId;
    }
};

} // namespace gpu
