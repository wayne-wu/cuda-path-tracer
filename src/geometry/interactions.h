#pragma once

#include <cuda_runtime.h>
#include <thrust/random.h>

#include "intersections.h"


__host__ __device__ inline float srgbToLinear(float c) {
    c = glm::clamp(c, 0.0f, 1.0f);
    return c <= 0.04045f ? c / 12.92f : powf((c + 0.055f) / 1.055f, 2.4f);
}

__host__ __device__ inline Color srgbToLinear(const Color& c) {
    return Color(srgbToLinear(c.r), srgbToLinear(c.g), srgbToLinear(c.b));
}

__host__ __device__ inline float linearToSrgb(float c) {
    c = glm::clamp(c, 0.0f, 1.0f);
    return c <= 0.0031308f ? 12.92f * c : 1.055f * powf(c, 1.0f / 2.4f) - 0.055f;
}

__host__ __device__ inline Color linearToSrgb(const Color& c) {
    return Color(linearToSrgb(c.r), linearToSrgb(c.g), linearToSrgb(c.b));
}

__device__ inline float DistributionGGX(const float NdotH, float roughness) {
    const float a = roughness * roughness;
    const float a2 = a * a;
    const float NdotH2 = NdotH * NdotH;

    const float denom = NdotH2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom);
}

__device__ inline float GeometrySchlickGGX(float NdotV, float roughness) {
    const float r = roughness + 1.0f;
    const float k = (r * r) / 8.0f;

    const float denom = NdotV * (1.0f - k) + k;
    return NdotV / denom;
}

__device__ inline glm::vec3 fresnelSchlick(float cosTheta, const glm::vec3& F0) {
    return F0 + (glm::vec3(1.0f) - F0) * powf(1.0f - cosTheta, 5.0f);
}

__device__ inline glm::vec3 microfacetBRDF(
    const glm::vec3& N,
    const glm::vec3& V,
    const glm::vec3& L,
    const glm::vec3& albedo,
    const float metallic,
    const float roughness) {
    const glm::vec3 H = glm::normalize(V + L);
    const float NdotL = max(glm::dot(N, L), 0.0f);
    const float NdotV = max(glm::dot(N, V), 0.0f);
    const float NdotH = max(glm::dot(N, H), 0.0f);
    const float VdotH = max(glm::dot(V, H), 0.0f);

    const glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
    const glm::vec3 F = fresnelSchlick(VdotH, F0);
    const float D = DistributionGGX(NdotH, roughness);
    const float G = GeometrySchlickGGX(NdotV, roughness) * GeometrySchlickGGX(NdotL, roughness);

    return (D * G * F) / (4.0f * NdotV * NdotL + 1e-8f);
}

__device__ inline glm::vec3 diffuseBRDF(const glm::vec3& albedo) {
    return albedo / PI;
}

__device__ inline glm::vec3 evaluateBRDF(
    const glm::vec3& N,
    const glm::vec3& V,
    const glm::vec3& L,
    const glm::vec3& albedo,
    const float metallic,
    const float roughness) {
    const glm::vec3 H = glm::normalize(V + L);
    const glm::vec3 F0 = glm::mix(glm::vec3(0.04f), albedo, metallic);
    const glm::vec3 F = fresnelSchlick(max(glm::dot(V, H), 0.0f), F0);
    const glm::vec3 diffuse = (glm::vec3(1.0f) - F) * (1.0f - metallic) * diffuseBRDF(albedo);
    return diffuse + microfacetBRDF(N, V, L, albedo, metallic, roughness);
}

__device__ inline float diffusePDF(const float NdotL) {
    return max(NdotL, 0.0f) / PI;
}

__device__ inline float microfacetPDF(const glm::vec3& N, const glm::vec3& V, const glm::vec3& L, const float roughness) {
    const glm::vec3 H = glm::normalize(V + L);
    const float NdotH = max(glm::dot(N, H), 0.0f);
    const float VdotH = max(glm::dot(V, H), 0.0f);
    return DistributionGGX(NdotH, roughness) * NdotH / (4.0f * VdotH + 1e-8f);
}

__device__ glm::vec3 sampleGGXHalfVector(const glm::vec3& normal, float roughness, thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    const float u1 = u01(rng);
    const float u2 = u01(rng);

    const float a = roughness * roughness;
    const float phi = 2.0f * PI * u1;
    const float cosTheta = sqrt((1.0f - u2) / (1.0f + (a * a - 1.0f) * u2));
    const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    glm::vec3 sample(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta);

    glm::vec3 tangentDirection;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        tangentDirection = glm::normalize(glm::cross(normal, glm::vec3(1, 0, 0)));
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        tangentDirection = glm::normalize(glm::cross(normal, glm::vec3(0, 1, 0)));
    } else {
        tangentDirection = glm::normalize(glm::cross(normal, glm::vec3(0, 0, 1)));
    }
    glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangentDirection));

    glm::mat3 sampleToLocal(tangentDirection, bitangent, normal);
    return glm::normalize(sampleToLocal * sample);
}

__device__ glm::vec3 sampleCosineHemisphere(
    const glm::vec3& normal,
    thrust::default_random_engine& rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng));
    float over = sqrt(1 - up * up);
    float around = u01(rng) * TWO_PI;

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__device__ glm::vec3 calculateImperfectSpecularDirection(
    const glm::vec3& surfaceNormal,
    const glm::vec3& reflectedDirection,
    const glm::vec4& tangent,
    thrust::default_random_engine& rng,
    float roughness) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    const float a = roughness * roughness;
    const float phi = 2.0f * PI * u01(rng);
    const float cosTheta = sqrt((1.0f - u01(rng)) / (1.0f + (a * a - 1.0f) * u01(rng)));
    const float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    glm::vec3 sample(
        sinTheta * cos(phi),
        sinTheta * sin(phi),
        cosTheta);

    glm::mat3 sampleToLocal;
    sampleToLocal[2] = reflectedDirection;
    sampleToLocal[0] = glm::vec3(tangent);
    sampleToLocal[1] = glm::cross(sampleToLocal[2], sampleToLocal[0]);

    glm::vec3 dir = glm::normalize(sampleToLocal * sample);
    return glm::dot(dir, surfaceNormal) < 0.0f ? reflectedDirection : dir;
}

__device__ void normalMapping(glm::vec3& surfaceNormal, const glm::vec3& tangentSpaceNormal, const glm::vec4& tangent) {
    const glm::vec3 tangentDirection = glm::normalize(glm::vec3(tangent));
    const glm::vec3 bitangent = glm::normalize(glm::cross(surfaceNormal, tangentDirection)) * tangent.w;
    const glm::mat3 TBN(tangentDirection, bitangent, surfaceNormal);
    surfaceNormal = glm::normalize(TBN * tangentSpaceNormal);
}

__device__ Color checkerBoard(glm::vec3 pos) {
    const float scale = 20.0f;
    const int x = static_cast<int>(floorf(pos.x * scale));
    const int z = static_cast<int>(floorf(pos.z * scale));
    return ((x + z) & 1) ? Color(0.05f, 0.05f, 0.05f) : Color(0.8f, 0.8f, 0.8f);
}

__device__ Color sampleTexture(cudaTextureObject_t texObj, const glm::vec2 uv) {
    const float4 color = tex2D<float4>(texObj, uv.x, uv.y);
    return Color(color.x, color.y, color.z);
}
