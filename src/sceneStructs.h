#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "tiny_gltf.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))
#define NEAR 1.f
#define FAR 50.f


typedef int Offset;

typedef glm::vec2 Vec2;
typedef glm::vec3 Vec3;
typedef glm::vec4 Vec4;

typedef glm::vec3 Color;
typedef uint16_t Index;

typedef std::vector<Vec3> Vertices;
typedef std::vector<Index> Indices;

enum GeomType {
    SPHERE,
    CUBE,
    MESH,
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
    glm::vec3 inv_dir;
};

struct Mesh {
    int prim_count;
    int prim_offset;
};

struct Primitive {
    // number of points
    int count;
    int num_vertices;
    Offset i_offset;
    Offset v_offset;
    Offset bin_offset = -1;
    Offset bf_offset = -1;
    Offset n_offset = -1;
    Offset uv_offset = -1;
    Offset t_offset = -1;
    int mat_id;
    glm::vec3 bbox_max;
    glm::vec3 bbox_min;
    glm::mat4 pivot_xform;
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

struct PrimData {
    Primitive* primitives;
    glm::vec3* vertices;
    glm::vec3* normals;
    uint16_t* indices;
    glm::vec2* uvs;
    glm::vec4* tangents;

    //octree
    Bin* bins;
    int* binFaces;

    void free() {
      cudaFree(primitives);
      cudaFree(vertices);
      cudaFree(normals);
      cudaFree(indices);
      cudaFree(uvs);
      cudaFree(tangents);
      
      if (bins) cudaFree(bins);
      if (binFaces) cudaFree(binFaces);
    }
};

struct Geom {
    enum GeomType type;
    int materialid;
    int meshid;
    int padding;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;
};

struct Texture {
  int width;
  int height;
  int components;
  unsigned char* image;
  int size;
};

struct TextureInfo {
  int index = -1;
  int texCoord;

  TextureInfo& operator=(const tinygltf::TextureInfo t) {
    index = t.index;
    texCoord = t.texCoord;
    return *this;
  }
};

struct NormalTextureInfo : TextureInfo {
  float scale = 1.0f;

  NormalTextureInfo& operator=(const tinygltf::NormalTextureInfo t) {
    index = t.index;
    texCoord = t.texCoord;
    scale = t.scale;
    return *this;
  }
};

// Based on tinygltf::PbrMetallicRoughness
// pbrMetallicRoughness class defined in glTF 2.0 spec.
// Defining a custom PbrMetallicRoughness struct here because
// all vectors have to use glm::vec to be supported in CUDA
struct PbrMetallicRoughness {
  Color baseColorFactor = Color(1.0f);  // Change to vec4 if alpha is used
  TextureInfo baseColorTexture;
  float metallicFactor;   // default 1
  float roughnessFactor;  // default 1
  TextureInfo metallicRoughnessTexture;

  __host__ __device__ PbrMetallicRoughness()
    : baseColorFactor(Color(1.0f)),
      metallicFactor(1.0f),
      roughnessFactor(1.0f) {}
};

struct Material {
    int tex_offset = 0;

    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    bool gltf = false;

    PbrMetallicRoughness pbrMetallicRoughness;

    NormalTextureInfo normalTexture;
    TextureInfo emissiveTexture;

    //tinygltf::OcclusionTextureInfo occlusionTexture;
    //tinygltf::TextureInfo emissiveTexture;
    glm::vec3 emissiveFactor;  // length 3. default [0, 0, 0]
};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;
    glm::mat4 viewMat;
    glm::mat4 projMat;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;
    int spp;
};

struct PathSegment {
    Ray ray;
    Color color;
    int pixelIndex;
    int remainingBounces;
};

struct Hit {
  int geomId = -1;
  int meshId = -1;
  int primId = -1;
  int binId = -1;
  int faceId = -1;
  Vec3 bary;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
  float t;
  int materialId;
  glm::vec3 surfaceNormal;
  glm::vec2 uv;
  glm::vec4 tangent;
  Hit hit;
  int padding;
};

struct GBufferPixel {
  float t;
  glm::vec3 n;  //normal
  glm::vec3 p;  //position
  float z;      //z-depth
};


// Predicate for checking if a path is complete or not
struct isPathCompleted {
  __host__ __device__
    bool operator()(const PathSegment& pathSegment) {
    return pathSegment.remainingBounces <= 0;
  }
};

struct compareIntersections {
  __host__ __device__
    bool operator()(const ShadeableIntersection& a, const ShadeableIntersection& b) {
    return a.materialId > b.materialId;
  }
};