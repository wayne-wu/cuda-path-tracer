#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "glm/glm.hpp"

enum class RenderGeometryType {
    Sphere,
    Cube,
    Mesh,
};

struct RenderBin {
    glm::vec3 bboxMin = glm::vec3(0.0f);
    glm::vec3 bboxMax = glm::vec3(0.0f);
    int childIndex = -1;
    int startIndex = -1;
    int endIndex = -1;
    int depth = 0;
};

struct RenderMesh {
    std::vector<uint32_t> indices;
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> uvs;
    std::vector<glm::vec4> tangents;
    std::vector<RenderBin> bins;
    std::vector<int> binFaces;
};

struct RenderInstance {
    RenderGeometryType geometryType = RenderGeometryType::Mesh;
    int meshId = -1;
    int materialId = -1;

    glm::mat4 transform = glm::mat4(1.0f);
    glm::mat4 inverseTransform = glm::mat4(1.0f);
    glm::mat4 inverseTranspose = glm::mat4(1.0f);

    glm::vec3 bboxMin = glm::vec3(0.0f);
    glm::vec3 bboxMax = glm::vec3(0.0f);
};

struct RenderTextureRef {
    int textureId = -1;
    int texCoord = 0;

    bool valid() const { return textureId >= 0; }
};

struct RenderMaterial {
    glm::vec3 baseColor = glm::vec3(1.0f);
    RenderTextureRef baseColorTexture;

    float metallic = 0.0f;
    float roughness = 0.5f;
    RenderTextureRef metallicRoughnessTexture;

    glm::vec3 emissiveColor = glm::vec3(0.0f);
    float emissiveStrength = 0.0f;
    RenderTextureRef emissiveTexture;

    RenderTextureRef normalTexture;
    float normalScale = 1.0f;

    float transmission = 0.0f;
    float ior = 1.5f;

    float alpha = 1.0f;
    bool doubleSided = false;
    bool useRoughSpecular = false;
};

struct RenderTexture {
    int width = 0;
    int height = 0;
    int components = 0;
    std::vector<unsigned char> pixels;
};

enum class RenderCameraProjection {
    Perspective,
    Orthographic
};

struct RenderCamera {
    RenderCameraProjection projection = RenderCameraProjection::Perspective;

    glm::ivec2 resolution = glm::ivec2(800, 800);

    glm::vec3 position = glm::vec3(0.0f);
    glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, -1.0f);
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);

    float verticalFovDegrees = 45.0f;
    float orthographicScale = 1.0f;

    float nearClip = 0.1f;
    float farClip = 10000.0f;

    float apertureRadius = 0.0f;
    float focalDistance = 1.0f;
};

struct RenderEnvironment {
    int width = 0;
    int height = 0;
    float intensity = 1.0f;
    float rotation = 0.0f;
    bool enabled = false;
    std::vector<glm::vec4> pixels;
};

struct RenderSettings {
    unsigned int iterations = 100;
    int traceDepth = 5;
    int spp = 1;
    std::string imageName;
};

struct RenderScene {
    std::vector<RenderMesh> meshes;
    std::vector<RenderInstance> instances;
    std::vector<RenderMaterial> materials;
    std::vector<RenderTexture> textures;
    RenderCamera camera;
    RenderEnvironment environment;
    RenderSettings settings;
};
