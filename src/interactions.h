#pragma once


#include "intersections.h"
#include "cuda_runtime.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__device__
glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));


    float c, s;
    __sincosf(around, &s, &c);

    return up * normal
        + c * over * perpendicularDirection1
        + s * over * perpendicularDirection2;
}

/**
 * Computes the imperfect specular ray direction.
 * Based on: https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-20-gpu-based-importance-sampling
 */
__device__
glm::vec3 calculateImperfectSpecularDirection(
    glm::vec3 normal, glm::vec3 reflect, glm::vec4 tangent,
    thrust::default_random_engine& rng, 
    float roughness) {
   
    thrust::uniform_real_distribution<float> u01(0,1);
    float x1 = u01(rng);

    float sp, cp, st, ct;
    __sincosf(2 * PI * u01(rng), &sp, &cp);
    __sincosf(atan2f(roughness * sqrtf(x1), sqrtf(1-x1)), &st, &ct);

    glm::vec3 dir;
    dir.x = cp * st;
    dir.y = sp * st;
    dir.z = ct;
    
    glm::mat3 worldToLocal;
    worldToLocal[2] = normal;
    worldToLocal[1] = glm::vec3(tangent);  // t
    worldToLocal[0] = glm::cross(normal, worldToLocal[1]) * tangent.w;  // b

    glm::vec3 r = glm::normalize(worldToLocal*reflect);
    
    /// construct an under-constrained coordinate using reflection as up axis
    glm::mat3 sampleToLocal;
    sampleToLocal[2] = r;
    sampleToLocal[0] = glm::normalize(glm::vec3(0, r.z, -r.y));
    sampleToLocal[1] = glm::cross(sampleToLocal[2], sampleToLocal[1]);

    glm::mat3 localToWorld = glm::inverse(worldToLocal);
    glm::mat3 sampleToWorld = localToWorld * sampleToLocal;

    dir = glm::normalize(sampleToWorld * dir);

    return dir;
}

/**
* Based on: https://learnopengl.com/Advanced-Lighting/Normal-Mapping 
*/
__device__
void normalMapping(glm::vec3& n, const glm::vec3& nMap, const glm::vec4& tangent) {
  glm::vec3 t = glm::vec3(tangent);
  glm::vec3 b = glm::cross(n, t) * tangent.w;
  glm::mat3 TBN = glm::mat3(t, b, n);
  n = glm::normalize(TBN * nMap);
}

/**
* Based on: https://pbr-book.org/3ed-2018/Texture/Solid_and_Procedural_Texturing
*/
__device__
Color checkerBoard(const glm::vec2 uv) {
  glm::vec2 v = 10.f * uv;  // scale UV based on the frequency
  return (int(v.x) + int(v.y)) % 2 == 0 ? Color(1.f) : Color(0.f);
}

/**
* Helper function for sampling the texture object with the given UV coordinate
*/
__device__
Color sampleTexture(cudaTextureObject_t texObj, const glm::vec2 uv) {
  // NOTE: cudaReadModeNormalizedFloat will convert uchar4 to float4
  float4 rgba = tex2D<float4>(texObj, uv.x, uv.y);
  return Color(rgba.x, rgba.y, rgba.z);
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */
__device__
void scatterRay(
    PathSegment& pathSegment,
    const ShadeableIntersection& i,
    const Material& m,
    cudaTextureObject_t* textures,
    thrust::default_random_engine &rng) {
    
    glm::vec3 intersect = getPointOnRay(pathSegment.ray, i.t);

    Color color;  // Final Color
    glm::vec3 newDir;  // New Ray Direction

    // PBR Material Properties
    Color albedo;  // baseColor
    float pM, pR;  // Metallic and Roughness Parameters
    float ao;  // TODO: Ambient Occlusion
    glm::vec3 normal = i.surfaceNormal;
 
    // Get Albedo
    int txId = -1;
    txId = m.tex_offset + m.pbrMetallicRoughness.baseColorTexture.index;
    if (txId < 0) {
      albedo = m.pbrMetallicRoughness.baseColorFactor;
    }
    else {
      albedo = sampleTexture(textures[txId], i.uv);
      //albedo = checkerBoard(i.uv);
    }

    // Get Metallic and Roughness Parameters
    txId = m.tex_offset + m.pbrMetallicRoughness.metallicRoughnessTexture.index;
    if (txId < 0) {
      pM = m.pbrMetallicRoughness.metallicFactor;
      pR = m.pbrMetallicRoughness.roughnessFactor;
    }
    else {
      Color pbr = sampleTexture(textures[txId], i.uv);
      pM = pbr.b * m.pbrMetallicRoughness.metallicFactor;
      pR = pbr.g * m.pbrMetallicRoughness.roughnessFactor;
    }

    // Apply normal map if any
    txId = m.tex_offset + m.normalTexture.index;
    if (txId >= 0) {
      glm::vec3 n = sampleTexture(textures[txId], i.uv);
      n = glm::normalize(n * 2.f - 1.f);
      normalMapping(normal, n, i.tangent);
    }

    thrust::uniform_real_distribution<float> u01(0, 1);

    if (u01(rng) < pM) {
      // Specular
      glm::vec3 reflect = glm::reflect(pathSegment.ray.direction, normal);
      newDir = m.gltf ? calculateImperfectSpecularDirection(normal, reflect, i.tangent, rng, pR) : reflect;
      // newDir = reflect;  //<---- use this for perfect specular
      color = pM * albedo;
    }
    else {
      // Diffuse
      newDir = calculateRandomDirectionInHemisphere(normal, rng);
      color = (1.f - pM) * albedo;
    }

    pathSegment.ray.origin = intersect + (newDir * 0.0001f);;
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.color *= color;
}


