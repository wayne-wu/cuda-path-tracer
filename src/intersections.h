#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"

/**
 * Swap the given two vectors
 */
__host__ __device__ inline void swapVal(float& v1, float& v2) {
  float tmp = v1;
  v1 = v2;
  v2 = tmp;
}

/**
 * Interpolate a vector given three vectors and a barycentric coordinate (uv)
 * TODO: Fast lerp: https://developer.nvidia.com/blog/lerp-faster-cuda/ 
 */
template<class T>
__host__ __device__ inline void lerp(T& p, const T& a, 
  const T& b, const T& c, const float u, const float v) {
    p = (1.0f - u - v) * a + u * b + v * c;
}

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom& box, Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom& sphere, Ray& r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

/**
 * Test intersection between a ray and a box given the min and max points (bbox)
 * TODO: https://tavianator.com/2022/ray_box_boundary.html 
 */
__host__ __device__ bool intersectBox(const Ray& r, const glm::vec3& v_min, const glm::vec3& v_max) {

    const glm::vec3& ro = r.origin;
    const glm::vec3& rd = r.direction;
    const glm::vec3& n = r.inv_dir;

    float tmin, tmax;

    float tx1 = (v_min.x - ro.x) * n.x;
    float tx2 = (v_max.x - ro.x) * n.x;

    tmin = min(tx1, tx2);
    tmax = max(tx1, tx2);

    float ty1 = (v_min.y - ro.y) * n.y;
    float ty2 = (v_max.y - ro.y) * n.y;

    tmin = min(tmin, min(ty1, ty2));
    tmax = max(tmax, max(ty1, ty2));

    float tz1 = (v_min.z - ro.z) * n.z;
    float tz2 = (v_max.z - ro.z) * n.z;

    tmin = min(tmin, min(tz1, tz2));
    tmax = max(tmax, max(tz1, tz2));

    return tmax >= tmin;
}

__host__ __device__ bool intersectFace(const PrimData& md, const Primitive& m, const Ray& r,
                              int faceIdx, Vec3& bary) {
  int i = 3 * faceIdx;
  return glm::intersectRayTriangle(r.origin, r.direction,
    md.vertices[m.v_offset + md.indices[m.i_offset + i++]],
    md.vertices[m.v_offset + md.indices[m.i_offset + i++]],
    md.vertices[m.v_offset + md.indices[m.i_offset + i]], bary);
}


/**
 * Test intersection between a ray and a mesh
 */
__host__ __device__ bool meshIntersectionTest(Geom& geom, Mesh& mesh, PrimData& md, Ray& r, ShadeableIntersection& intersection){

    Vec3 rayOrigin = r.origin;

    r.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    r.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    r.inv_dir = 1.0f / r.direction;

    int hitPrim = -1;
    int hitFace = -1;
    glm::vec3 hitBary, bary;
    hitBary.z = intersection.t;  // FLT_MAX

    for (int primId = 0; primId < mesh.prim_count; primId++) {

      const Primitive m = md.primitives[mesh.prim_offset + primId];

      // Try intersecting with the bbox first
      if (!intersectBox(r, m.bbox_min, m.bbox_max))
        continue;

      if (m.bin_offset >= 0) {

        const Offset& bo = m.bin_offset;

        // Use octree for intersection testing
        int stack[64];
        int* stackPtr = stack;
        *stackPtr++ = -1;

        int bin = 0;

        do {

          glm::vec3 minCorner = md.binCorners[bo + 2 * bin];
          glm::vec3 maxCorner = md.binCorners[bo + 2 * bin + 1];
          if (intersectBox(r, minCorner, maxCorner)) {
            // Check if the current bin is leaf
            if (md.binChildIndices[bo + bin] == -1) {
              int startIdx = md.binStartIndices[bo + bin];
              int endIdx = md.binEndIndices[bo + bin];

              if (startIdx >= 0) {
                for (int b = startIdx; b < endIdx; ++b) {
                  int faceIdx = md.faceBins[m.bf_offset + b];
                  if (intersectFace(md, m, r, faceIdx, bary) && bary.z < hitBary.z) {
                    hitPrim = primId;
                    hitFace = faceIdx;
                    hitBary = bary;
                  }
                }
              }
            }
            else {
              int childBin = md.binChildIndices[bo + bin];
              for (int i = childBin; i < childBin + 8; ++i) {
                *stackPtr++ = i;  // push children bins to stack
              }
            }
          }

          bin = *--stackPtr;  // pop from stack

        } while (bin >= 0);
      }
      else
      {
        // Test intersection on all triangles in the mesh
        int numFaces = m.count / 3;
        for (int i = 0; i < numFaces; ++i) {
          if (intersectFace(md, m, r, i, bary) && bary.z < hitBary.z) {
            hitPrim = primId;
            hitFace = i;
            hitBary = bary;
          }
        }
      }
    }

    // TODO: Reduce divergence. Maybe move everything below to a new kernel after compaction?
    if (hitPrim >= 0) {

      const Primitive m = md.primitives[mesh.prim_offset + hitPrim];

      int f0 = md.indices[m.i_offset + 3 * hitFace];
      int f1 = md.indices[m.i_offset + 3 * hitFace + 1];
      int f2 = md.indices[m.i_offset + 3 * hitFace + 2];

      glm::vec3 v0, v1, v2;
      v0 = md.vertices[m.v_offset + f0];
      v1 = md.vertices[m.v_offset + f1];
      v2 = md.vertices[m.v_offset + f2];

      // Interpolate Normal
      if (m.n_offset >= 0) {
        lerp<glm::vec3>(intersection.surfaceNormal, 
          md.normals[m.n_offset + f0], 
          md.normals[m.n_offset + f1], 
          md.normals[m.n_offset + f2], hitBary.x, hitBary.y);
      }

      // Interpolate UV
      glm::vec2 uv0, uv1, uv2;
      if (m.uv_offset >= 0) {
        uv0 = md.uvs[m.uv_offset + f0];
        uv1 = md.uvs[m.uv_offset + f1];
        uv2 = md.uvs[m.uv_offset + f2];
        lerp<glm::vec2>(intersection.uv, uv0, uv1, uv2, hitBary.x, hitBary.y);
      }

      // Interpolate Tangent
      if (m.t_offset >= 0) {
        lerp<glm::vec4>(intersection.tangent, 
          md.tangents[m.t_offset + f0], 
          md.tangents[m.t_offset + f1], 
          md.tangents[m.t_offset + f2], 
          hitBary.x, hitBary.y);
      }
      else {
        // Calculate tangent vector: 
        // https://www.cs.upc.edu/~virtual/G/1.%20Teoria/06.%20Textures/Tangent%20Space%20Calculation.pdf

        glm::vec3 dp1 = v1 - v0;
        glm::vec3 dp2 = v2 - v0;
        glm::vec2 du1 = uv1 - uv0;
        glm::vec2 du2 = uv2 - uv0;

        float r = 1.0F / (du1.x * du2.y - du2.x * du1.y);
        glm::vec3 sdir((du2.y * dp1.x - du1.y * dp2.x) * r, (du2.y * dp1.y - du1.y * dp2.y) * r,
          (du2.y * dp1.z - du1.y * dp2.z) * r);
        glm::vec3 tdir((du1.x * dp2.x - du2.x * dp1.x) * r, (du1.x * dp2.y - du2.x * dp1.y) * r,
          (du1.x * dp2.z - du2.x * dp1.z) * r);

        intersection.tangent = glm::vec4(
          glm::normalize(sdir - intersection.surfaceNormal * glm::dot(intersection.surfaceNormal, sdir)),
          glm::dot(glm::cross(intersection.surfaceNormal, sdir), tdir) < 0.f ? -1.f : 1.f);
      }

      intersection.surfaceNormal = glm::normalize(
        multiplyMV(geom.invTranspose, glm::vec4(intersection.surfaceNormal, 1.f)));
      intersection.tangent = glm::vec4(glm::normalize(
        multiplyMV(geom.invTranspose, intersection.tangent)), intersection.tangent.z);

      intersection.materialId = m.mat_id;

      intersection.t = glm::length(rayOrigin - multiplyMV(geom.transform, glm::vec4(getPointOnRay(r, hitBary.z), 1.f)));

      return true;
    }

    return false;
}