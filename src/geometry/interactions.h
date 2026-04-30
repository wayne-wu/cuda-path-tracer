#pragma once

#include <cuda_runtime.h>
#include <thrust/random.h>

#include "intersections.h"

__device__ glm::vec3 calculateRandomDirectionInHemisphere(
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
