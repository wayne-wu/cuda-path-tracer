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

/**
 * Multiplies a mat4 and a vec3 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV1(glm::mat4 m, glm::vec3 v) {
  return glm::vec3(m * glm::vec4(v, 1.f));
}

__host__ __device__ glm::vec3 multiplyMV0(glm::mat4 m, glm::vec3 v) {
  return glm::vec3(m * glm::vec4(v, 0.f));
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
__host__ __device__ bool intersectBox(const Ray& r, const glm::vec3& v_min, const glm::vec3& v_max, float& t) {

    const glm::vec3& ro = r.origin;
    const glm::vec3& n = r.inv_dir;

    Vec3 tmin = (v_min - ro) * n;
    Vec3 tmax = (v_max - ro) * n;
    Vec3 t1 = min(tmin, tmax);
    Vec3 t2 = max(tmin, tmax);

    // tNear;
    t = max(max(t1.x, t1.y), t1.z);
    
    return t <= min(min(t2.x, t2.y), t2.z);
}

__host__ __device__ bool intersectFace(const PrimData& md, const Primitive& m, const Ray& r,
                              int faceIdx, Vec3& bary) {
  int i = 3 * faceIdx;
  return glm::intersectRayTriangle(r.origin, r.direction,
    md.vertices[m.v_offset + md.indices[m.i_offset + i++]],
    md.vertices[m.v_offset + md.indices[m.i_offset + i++]],
    md.vertices[m.v_offset + md.indices[m.i_offset + i]], bary);
}

/*
Compute the info needed for shading for a face/triangle.
*/
__host__ __device__ void computeFaceInfo(const PrimData& md, const Primitive& m, const int faceId, const Vec3& bary, 
                                         Vec3& normal, Vec2& uv, Vec4& tangent) {
  
  const int& hitFace = faceId;
  int f0 = md.indices[m.i_offset + 3 * hitFace];
  int f1 = md.indices[m.i_offset + 3 * hitFace + 1];
  int f2 = md.indices[m.i_offset + 3 * hitFace + 2];

  glm::vec3 v0, v1, v2;
  v0 = md.vertices[m.v_offset + f0];
  v1 = md.vertices[m.v_offset + f1];
  v2 = md.vertices[m.v_offset + f2];

  // Interpolate Normal
  if (m.n_offset >= 0) {
    lerp<glm::vec3>(normal,
      md.normals[m.n_offset + f0],
      md.normals[m.n_offset + f1],
      md.normals[m.n_offset + f2], bary.x, bary.y);
  }

  // Interpolate UV
  glm::vec2 uv0, uv1, uv2;
  if (m.uv_offset >= 0) {
    uv0 = md.uvs[m.uv_offset + f0];
    uv1 = md.uvs[m.uv_offset + f1];
    uv2 = md.uvs[m.uv_offset + f2];
    lerp<glm::vec2>(uv, uv0, uv1, uv2, bary.x, bary.y);
  }

  // Interpolate Tangent
  if (m.t_offset >= 0) {
    lerp<glm::vec4>(tangent,
      md.tangents[m.t_offset + f0],
      md.tangents[m.t_offset + f1],
      md.tangents[m.t_offset + f2],
      bary.x, bary.y);
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

    tangent = glm::vec4(
      glm::normalize(sdir - normal * glm::dot(normal, sdir)),
      glm::dot(glm::cross(normal, sdir), tdir) < 0.f ? -1.f : 1.f);
  }
}


/**
 * Test intersection between a ray and a mesh
 */
__host__ __device__ bool meshIntersectionTest(const Geom& geom, const Mesh& mesh, Primitive* prims, const PrimData& md, Ray& r, ShadeableIntersection& intersection){

    Vec3 rayOrigin = r.origin;

    Vec3 ptWS = getPointOnRay(r, intersection.t);

    // Move ray into object space
    r.origin = multiplyMV(geom.inverseTransform, glm::vec4(r.origin, 1.0f));
    r.direction = glm::normalize(multiplyMV(geom.inverseTransform, glm::vec4(r.direction, 0.0f)));
    r.inv_dir = 1.0f / r.direction;

    glm::vec3 bary;
    glm::vec3& hitBary = intersection.hit.bary;

    // Get the current closest distance in object space
    hitBary.z = intersection.hit.geomId < 0 ? FLT_MAX : 
      glm::length(r.origin - multiplyMV1(geom.inverseTransform, ptWS));

    for (int primId = mesh.prim_offset; primId < mesh.prim_offset + mesh.prim_count; ++primId) {

      const Primitive& m = prims[primId];

      if (m.bin_offset >= 0) {
        
        Bin binInfo;
        const Offset& bo = m.bin_offset;

        // Use octree for intersection testing
        int stack[64];
        int* stackPtr = stack;
        *stackPtr++ = -1;

        int bin = 0;

        do {
          binInfo = md.bins[bo + bin];

          if (intersectBox(r, binInfo.bbox_min, binInfo.bbox_max, bary.z) && bary.z < hitBary.z) {
            if (binInfo.childIndex != -1)  // check if has more children
              for (int i = binInfo.childIndex; i < binInfo.childIndex + 8; ++i) {
                *stackPtr++ = i;  // push children bins to stack
              }
            else if (binInfo.startIndex >= 0) { 
              for (int b = binInfo.startIndex; b < binInfo.endIndex; ++b) {
                int faceIdx = md.binFaces[m.bf_offset + b];
                if (intersectFace(md, m, r, faceIdx, bary) && bary.z < hitBary.z) {
                  intersection.hit.primId = primId;
                  intersection.hit.faceId = faceIdx;
                  intersection.hit.bary = bary;
                }
              }
            }
          }

          bin = *--stackPtr;  // pop from stack

        } while (bin >= 0);
      }
      else
      {
        // Try intersecting with the bbox first
        if (!intersectBox(r, m.bbox_min, m.bbox_max, bary.z))
          continue;

        // Test intersection on all triangles in the mesh
        int numFaces = m.count / 3;
        for (int i = 0; i < numFaces; ++i) {
          if (intersectFace(md, m, r, i, bary) && bary.z < hitBary.z) {
            intersection.hit.primId = primId;
            intersection.hit.faceId = i;
            intersection.hit.bary = bary;
          }
        }
      }
    }

    // TODO: Reduce divergence. Maybe move everything below to a new kernel after compaction?
    if (intersection.hit.primId >= 0) {

      const Primitive& m = prims[intersection.hit.primId];

      computeFaceInfo(md, m, intersection.hit.faceId, intersection.hit.bary,
        intersection.surfaceNormal, intersection.uv, intersection.tangent);

      intersection.surfaceNormal = glm::normalize(
        multiplyMV0(geom.invTranspose, intersection.surfaceNormal));
      intersection.tangent = glm::vec4(
        glm::normalize(multiplyMV0(geom.invTranspose, glm::vec3(intersection.tangent))), 
        intersection.tangent.w);

      intersection.materialId = m.mat_id;

      // Calculate intersection distance in world space
      intersection.t = glm::length(rayOrigin - multiplyMV1(geom.transform, getPointOnRay(r, intersection.hit.bary.z)));

      return true;
    }

    return false;
}