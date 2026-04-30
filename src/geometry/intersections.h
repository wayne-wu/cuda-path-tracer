#pragma once

#include <cstdint>

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "core/utilities.h"
#include "scene/GpuSceneTypes.h"

using namespace gpu;

template <class T>
__host__ __device__ inline void lerp(T& p, const T& a, const T& b, const T& c, float u, float v) {
    p = (1.0f - u - v) * a + u * b + v * c;
}

__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

__host__ __device__ inline Vec3 getPointOnRay(const Ray& r, float t) {
    return r.origin + (t - 0.0001f) * glm::normalize(r.direction);
}

__host__ __device__ inline Vec3 multiplyMV1(const Mat4& m, const Vec3& v) {
    return Vec3(m * Vec4(v, 1.0f));
}

__host__ __device__ inline Vec3 multiplyMV0(const Mat4& m, const Vec3& v) {
    return Vec3(m * Vec4(v, 0.0f));
}

__host__ __device__ inline Vec3 getBoxNormal(const Vec3& p) {
    const Vec3 absP = glm::abs(p);
    if (absP.x >= absP.y && absP.x >= absP.z) {
        return Vec3(p.x < 0.0f ? -1.0f : 1.0f, 0.0f, 0.0f);
    }
    if (absP.y >= absP.z) {
        return Vec3(0.0f, p.y < 0.0f ? -1.0f : 1.0f, 0.0f);
    }
    return Vec3(0.0f, 0.0f, p.z < 0.0f ? -1.0f : 1.0f);
}

__host__ __device__ inline bool intersectBox(const Ray& r, const Vec3& vMin, const Vec3& vMax, float& t) {
    const Vec3 tmin = (vMin - r.origin) * r.inv_dir;
    const Vec3 tmax = (vMax - r.origin) * r.inv_dir;
    const Vec3 t1 = glm::min(tmin, tmax);
    const Vec3 t2 = glm::max(tmin, tmax);

    t = glm::max(glm::max(t1.x, t1.y), t1.z);
    return t <= glm::min(glm::min(t2.x, t2.y), t2.z);
}

__device__ inline Ray getRayOS(const gpu::Instance& instance, const Ray& rayWS) {
    Ray rayOS;
    rayOS.origin = multiplyMV1(instance.inverseTransform, rayWS.origin);
    rayOS.direction = glm::normalize(multiplyMV0(instance.inverseTransform, rayWS.direction));
    rayOS.inv_dir = 1.0f / rayOS.direction;
    return rayOS;
}

__device__ inline Vec3 makePerpendicularTangent(const Vec3& normal) {
    const Vec3 n = glm::normalize(normal);
    const Vec3 reference = glm::abs(n.y) < 0.999f ? Vec3(0.0f, 1.0f, 0.0f) : Vec3(1.0f, 0.0f, 0.0f);
    return glm::normalize(glm::cross(reference, n));
}

__device__ inline Vec4 makeAnalyticTangent(const Vec3& normalOS, const Vec3& pointOS,
                                           gpu::GeometryType geometryType) {
    if (geometryType == gpu::GeometryType::Sphere) {
        Vec3 tangent(-pointOS.z, 0.0f, pointOS.x);
        if (glm::dot(tangent, tangent) > 1e-8f) {
            return Vec4(glm::normalize(tangent), 1.0f);
        }
    }
    return Vec4(makePerpendicularTangent(normalOS), 1.0f);
}

__device__ inline Vec4 transformTangentToWorld(const gpu::Instance& instance, const Vec3& normalWS,
                                               const Vec4& tangentOS) {
    Vec3 tangentWS = multiplyMV0(instance.transform, Vec3(tangentOS));
    tangentWS = tangentWS - normalWS * glm::dot(normalWS, tangentWS);
    if (glm::dot(tangentWS, tangentWS) <= 1e-8f) {
        tangentWS = makePerpendicularTangent(normalWS);
    }
    return Vec4(glm::normalize(tangentWS), tangentOS.w);
}

__device__ inline float boxIntersectionTest(const Ray& rayOS) {
    float tmin = -1e38f;
    float tmax = 1e38f;
    for (int xyz = 0; xyz < 3; ++xyz) {
        const float t1 = (-0.5f - rayOS.origin[xyz]) / rayOS.direction[xyz];
        const float t2 = (+0.5f - rayOS.origin[xyz]) / rayOS.direction[xyz];
        const float ta = glm::min(t1, t2);
        const float tb = glm::max(t1, t2);
        if (ta > 0.0f && ta > tmin) {
            tmin = ta;
        }
        if (tb < tmax) {
            tmax = tb;
        }
    }

    if (tmax >= tmin && tmax > 0.0f) {
        return tmin <= 0.0f ? tmax : tmin;
    }
    return -1.0f;
}

__device__ inline float sphereIntersectionTest(const Ray& rayOS) {
    const float radius = 0.5f;
    const float vDotDirection = glm::dot(rayOS.origin, rayOS.direction);
    const float radicand = vDotDirection * vDotDirection - (glm::dot(rayOS.origin, rayOS.origin) - radius * radius);
    if (radicand < 0.0f) {
        return -1.0f;
    }

    const float squareRoot = sqrtf(radicand);
    const float t1 = -vDotDirection + squareRoot;
    const float t2 = -vDotDirection - squareRoot;

    if (t1 < 0.0f && t2 < 0.0f) {
        return -1.0f;
    }
    if (t1 > 0.0f && t2 > 0.0f) {
        return glm::min(t1, t2);
    }
    return glm::max(t1, t2);
}

__device__ inline bool intersectFace(const gpu::PrimitiveDataView& data, const gpu::Primitive& primitive,
                                     const Ray& ray, int faceIndex, Vec3& bary) {
    const gpu::Triangle& triangle = data.triangles[primitive.face_offset + faceIndex];
    return glm::intersectRayTriangle(ray.origin, ray.direction, triangle.v0, triangle.v1, triangle.v2, bary);
}

__device__ inline float meshIntersectionTest(const Ray& rayOS, const gpu::Mesh& mesh,
                                             const gpu::SceneView& scene, Hit& hit) {
    float minT = -1.0f;
    for (int primIndex = 0; primIndex < mesh.prim_count; ++primIndex) {
        const int primitiveIndex = mesh.prim_offset + primIndex;
        const gpu::Primitive& primitive = scene.primData.primitives[primitiveIndex];

        if (primitive.bin_offset >= 0 && scene.primData.bins != nullptr && scene.primData.binFaces != nullptr) {
            const Offset binOffset = primitive.bin_offset;
            int stack[64];
            int* stackPtr = stack;
            *stackPtr++ = -1;
            int binIndex = 0;

            do {
                const gpu::Bin& bin = scene.primData.bins[binOffset + binIndex];
                float binT = -1.0f;
                if (intersectBox(rayOS, bin.bbox_min, bin.bbox_max, binT) &&
                    (minT < 0.0f || binT < minT || binT < 0.0f)) {
                    if (bin.childIndex != -1) {
                        int childBins[8];
                        float childTs[8];
                        int childCount = 0;

                        for (int childIndex = bin.childIndex; childIndex < bin.childIndex + 8; ++childIndex) {
                            const gpu::Bin& child = scene.primData.bins[binOffset + childIndex];
                            float childT = -1.0f;
                            if (intersectBox(rayOS, child.bbox_min, child.bbox_max, childT) &&
                                (minT < 0.0f || childT < minT || childT < 0.0f)) {
                                int insertPos = childCount;
                                while (insertPos > 0 && childTs[insertPos - 1] < childT) {
                                    childTs[insertPos] = childTs[insertPos - 1];
                                    childBins[insertPos] = childBins[insertPos - 1];
                                    --insertPos;
                                }
                                childTs[insertPos] = childT;
                                childBins[insertPos] = childIndex;
                                ++childCount;
                            }
                        }

                        for (int i = 0; i < childCount; ++i) {
                            *stackPtr++ = childBins[i];
                        }
                    }
                    else if (bin.startIndex >= 0) {
                        for (int b = bin.startIndex; b < bin.endIndex; ++b) {
                            const int face = scene.primData.binFaces[primitive.bf_offset + b];
                            Vec3 bary;
                            if (!intersectFace(scene.primData, primitive, rayOS, face, bary)) {
                                continue;
                            }

                            const float localT = bary.z;
                            if (localT > 0.0f && (minT < 0.0f || localT < minT)) {
                                minT = localT;
                                hit.primId = primitiveIndex;
                                hit.faceId = face;
                                hit.bary = bary;
                                hit.t = localT;
                            }
                        }
                    }
                }

                binIndex = *--stackPtr;
            } while (binIndex >= 0);
        }
        else {
            float boxT = -1.0f;
            if (!intersectBox(rayOS, primitive.bbox_min, primitive.bbox_max, boxT) ||
                (minT >= 0.0f && boxT >= minT && boxT >= 0.0f)) {
                continue;
            }

            for (int face = 0; face < primitive.count / 3; ++face) {
                Vec3 bary;
                if (!intersectFace(scene.primData, primitive, rayOS, face, bary)) {
                    continue;
                }

                const float localT = bary.z;
                if (localT > 0.0f && (minT < 0.0f || localT < minT)) {
                    minT = localT;
                    hit.primId = primitiveIndex;
                    hit.faceId = face;
                    hit.bary = bary;
                    hit.t = localT;
                }
            }
        }
    }
    return minT;
}

__device__ inline void computeFaceInfo(const gpu::PrimitiveDataView& data, const gpu::Primitive& primitive,
                                       int faceId, const Vec3& bary, Vec3& normal, Vec2& uv, Vec4& tangent) {
    const uint32_t f0 = data.indices[primitive.i_offset + 3 * faceId + 0];
    const uint32_t f1 = data.indices[primitive.i_offset + 3 * faceId + 1];
    const uint32_t f2 = data.indices[primitive.i_offset + 3 * faceId + 2];

    if (primitive.n_offset >= 0) {
        lerp(normal,
             data.normals[primitive.n_offset + f0],
             data.normals[primitive.n_offset + f1],
             data.normals[primitive.n_offset + f2],
             bary.x,
             bary.y);
    }

    Vec2 uv0(0.0f);
    Vec2 uv1(0.0f);
    Vec2 uv2(0.0f);
    if (primitive.uv_offset >= 0) {
        uv0 = data.uvs[primitive.uv_offset + f0];
        uv1 = data.uvs[primitive.uv_offset + f1];
        uv2 = data.uvs[primitive.uv_offset + f2];
        lerp(uv, uv0, uv1, uv2, bary.x, bary.y);
    }

    if (primitive.t_offset >= 0) {
        lerp(tangent,
             data.tangents[primitive.t_offset + f0],
             data.tangents[primitive.t_offset + f1],
             data.tangents[primitive.t_offset + f2],
             bary.x,
             bary.y);
        return;
    }

    const Vec3 v0 = data.vertices[primitive.v_offset + f0];
    const Vec3 v1 = data.vertices[primitive.v_offset + f1];
    const Vec3 v2 = data.vertices[primitive.v_offset + f2];
    if (primitive.n_offset < 0) {
        normal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
    }

    const Vec3 dp1 = v1 - v0;
    const Vec3 dp2 = v2 - v0;
    const Vec2 du1 = uv1 - uv0;
    const Vec2 du2 = uv2 - uv0;
    const float denom = du1.x * du2.y - du2.x * du1.y;
    if (glm::abs(denom) <= 1e-8f) {
        tangent = Vec4(makePerpendicularTangent(normal), 1.0f);
        return;
    }

    const float r = 1.0f / denom;
    const Vec3 sdir((du2.y * dp1.x - du1.y * dp2.x) * r,
                    (du2.y * dp1.y - du1.y * dp2.y) * r,
                    (du2.y * dp1.z - du1.y * dp2.z) * r);
    const Vec3 tdir((du1.x * dp2.x - du2.x * dp1.x) * r,
                    (du1.x * dp2.y - du2.x * dp1.y) * r,
                    (du1.x * dp2.z - du2.x * dp1.z) * r);

    tangent = Vec4(
        glm::normalize(sdir - normal * glm::dot(normal, sdir)),
        glm::dot(glm::cross(normal, sdir), tdir) < 0.0f ? -1.0f : 1.0f);
}
