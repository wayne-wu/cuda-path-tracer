#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/partition.h>
#include <vector>
#include <iostream>

#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "core/utilities.h"
#include "PathTracer.h"
#include "intersections.h"
#include "interactions.h"
#include "renderer/RenderSettings.h"
#include "gpu/CudaBuffer.h"
#include "gpu/CudaTexture2D.h"
#include "gpu/GpuScene.h"
#include "scene/GpuSceneTypes.h"
#include "gpu/GpuFramebuffer.h"

#include "OptixDenoiser.h"
#include "optixCheck.h"

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char *msg, const char *file, int line) {
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err) {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file) {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
    getchar();
#  endif
    exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

// Kernel that writes the image to the OpenGL PBO directly.
__global__
void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, const float4* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        const float4 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp(static_cast<int>(pix.x * 255.0f), 0, 255);
        color.y = glm::clamp(static_cast<int>(pix.y * 255.0f), 0, 255);
        color.z = glm::clamp(static_cast<int>(pix.z * 255.0f), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__
void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, gpu::GBufferView gBuffer, int mode) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);

        switch (mode) {
        case GBUFFER_TIME:
        case GBUFFER_POSITION: {
            const float4 src = gBuffer.albedo[index];
            glm::vec3 albedo = glm::clamp(glm::vec3(src.x, src.y, src.z), glm::vec3(0.0f), glm::vec3(1.0f));
            pbo[index].w = 0;
            pbo[index].x = static_cast<unsigned char>(albedo.x * 255.0f);
            pbo[index].y = static_cast<unsigned char>(albedo.y * 255.0f);
            pbo[index].z = static_cast<unsigned char>(albedo.z * 255.0f);
            break;
        }

        case GBUFFER_NORMAL:
            const float4 src = gBuffer.normal[index];
            glm::vec3 n(src.x, src.y, src.z);
            pbo[index].w = 0;
            pbo[index].x = abs((int)(n.x * 255.0));
            pbo[index].y = abs((int)(n.y * 255.0));
            pbo[index].z = abs((int)(n.z * 255.0));
            break;
        }
    }
}

static void optixDenoiserLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << std::endl;
}


static Camera makeKernelCamera(const RenderCamera& renderCamera) {
    Camera camera = {};
    camera.resolution = renderCamera.resolution;
    camera.position = renderCamera.position;
    camera.lookAt = renderCamera.lookAt;

    glm::vec3 view = renderCamera.lookAt - renderCamera.position;
    if (glm::length2(view) == 0.0f) {
        view = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    camera.view = glm::normalize(view);

    glm::vec3 up = renderCamera.up;
    if (glm::length2(up) == 0.0f) {
        up = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    up = glm::normalize(up);

    glm::vec3 right = glm::cross(camera.view, up);
    if (glm::length2(right) == 0.0f) {
        right = glm::cross(camera.view, glm::vec3(0.0f, 0.0f, 1.0f));
    }
    if (glm::length2(right) == 0.0f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    }
    camera.right = glm::normalize(right);
    camera.up = glm::normalize(glm::cross(camera.right, camera.view));

    const float fovy = renderCamera.verticalFovDegrees > 0.0f ? renderCamera.verticalFovDegrees : 45.0f;
    const float yscaled = tan(fovy * (PI / 180.0f));
    const float xscaled = (yscaled * static_cast<float>(camera.resolution.x)) / static_cast<float>(camera.resolution.y);
    const float fovx = (atan(xscaled) * 180.0f) / PI;
    camera.fov = glm::vec2(fovx, fovy);
    camera.pixelLength = glm::vec2(
        2.0f * xscaled / static_cast<float>(camera.resolution.x),
        2.0f * yscaled / static_cast<float>(camera.resolution.y));

    return camera;
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__
void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = Color(1.0);

        // TODO: implement antialiasing by jittering the ray
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
            );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}


// Traversal/search
__global__
void traceClosestHits(
    int num_paths,
    gpu::SceneView scene,
    PathSegment* pathSegments,
    Hit* hits) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths) {
        const PathSegment& pathSegment = pathSegments[path_index];
        Hit hit;

        // naive parse through global instances
        for (int i = 0; i < scene.instanceCount; i++) {
            const gpu::Instance& instance = scene.instances[i];
            Ray ray = getRayOS(instance, pathSegment.ray);
            Hit candidateHit = Hit();
            float t = -1.0f;
            if (instance.geometryType == gpu::GeometryType::Cube) {
                t = boxIntersectionTest(ray);
            }
            else if (instance.geometryType == gpu::GeometryType::Sphere) {
                t = sphereIntersectionTest(ray);
            }
            else if (instance.geometryType == gpu::GeometryType::Mesh && instance.meshId >= 0) {
                t = meshIntersectionTest(ray, scene.meshes[instance.meshId], scene, candidateHit);
            }

            if (t > 0.f) {
                const Vec3 pointOS = getPointOnRay(ray, t);
                const Vec3 pointWS = multiplyMV1(instance.transform, pointOS);
                const float worldT = glm::length(pathSegment.ray.origin - pointWS);

                if (hit.t < 0.0f || worldT < hit.t) {
                    hit.geomId = i;
                    hit.meshId = instance.meshId;
                    hit.t = worldT;
                    hit.objectT = t;
                    hit.primId = candidateHit.primId;
                    hit.faceId = candidateHit.faceId;
                    hit.bary = candidateHit.bary;
                }
            }
        }

        hits[path_index] = hit;
    }
}

__global__
void resolveHitData(
    int num_paths,
    gpu::SceneView scene,
    PathSegment* pathSegments,
    Hit* hits,
    ShadeableIntersection* intersections) {
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index >= num_paths) {
        return;
    }

    const Ray& ray = pathSegments[path_index].ray;
    const Hit& hit = hits[path_index];
    ShadeableIntersection intersection;
    intersection.materialId = -1;
    intersection.surfaceNormal = glm::vec3(0.0f);
    intersection.uv = glm::vec2(0.0f);
    intersection.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
    intersection.t = -1.0f;

    if (hit.geomId < 0 || hit.t <= 0.0f) {
        intersections[path_index] = intersection;
        return;
    }

    const gpu::Instance& instance = scene.instances[hit.geomId];
    const Ray rayOS = getRayOS(instance, ray);
    const Vec3 pointOS = getPointOnRay(rayOS, hit.objectT);

    intersection.materialId = instance.materialId;
    if (instance.geometryType == gpu::GeometryType::Mesh) {
        const gpu::Primitive& prim = scene.primData.primitives[hit.primId];
        computeFaceInfo(
            scene.primData,
            prim,
            hit.faceId,
            hit.bary,
            intersection.surfaceNormal,
            intersection.uv,
            intersection.tangent);
    }
    else if (instance.geometryType == gpu::GeometryType::Cube) {
        intersection.surfaceNormal = getBoxNormal(pointOS);
        intersection.tangent = makeAnalyticTangent(intersection.surfaceNormal, pointOS, instance.geometryType);
    }
    else if (instance.geometryType == gpu::GeometryType::Sphere) {
        intersection.surfaceNormal = glm::normalize(pointOS);
        intersection.tangent = makeAnalyticTangent(intersection.surfaceNormal, pointOS, instance.geometryType);
    }

    intersection.surfaceNormal = glm::normalize(
        multiplyMV0(instance.inverseTranspose, intersection.surfaceNormal));
    intersection.tangent = transformTangentToWorld(instance, intersection.surfaceNormal, intersection.tangent);

    if (glm::dot(ray.direction, intersection.surfaceNormal) > 0.0f) {
        intersection.surfaceNormal = -intersection.surfaceNormal;
        intersection.tangent.w = -intersection.tangent.w;
    }

    intersection.t = hit.t;
    intersections[path_index] = intersection;
}

__device__
Color sampleEnvironment(
    cudaTextureObject_t envTex,
    const glm::vec3& direction,
    float intensity,
    float rotation) {
    glm::vec3 dir = glm::normalize(direction);
    float phi = atan2f(dir.z, dir.x) + rotation;
    float theta = acosf(glm::clamp(dir.y, -1.0f, 1.0f));
    float u = phi / TWO_PI + 0.5f;
    float v = theta / PI;

    float4 rgba = tex2D<float4>(envTex, u, v);
    return intensity * Color(rgba.x, rgba.y, rgba.z);
}

__device__
void scatterRay(
    PathSegment& pathSegment,
    const ShadeableIntersection& intersection,
    const gpu::Material& material,
    cudaTextureObject_t* textures,
    thrust::default_random_engine& rng) {
    const glm::vec3 intersect = getPointOnRay(pathSegment.ray, intersection.t);

    Color albedo = material.baseColor;
    if (material.baseColorTexture.index >= 0) {
        albedo = sampleTexture(textures[material.baseColorTexture.index], intersection.uv);
    }

    float metallic = material.metallic;
    float roughness = material.roughness;
    if (material.metallicRoughnessTexture.index >= 0) {
        const Color pbr = sampleTexture(textures[material.metallicRoughnessTexture.index], intersection.uv);
        metallic *= pbr.b;
        roughness *= pbr.g;
    }

    glm::vec3 normal = intersection.surfaceNormal;
    if (material.normalTexture.index >= 0) {
        glm::vec3 tangentSpaceNormal = sampleTexture(textures[material.normalTexture.index], intersection.uv);
        tangentSpaceNormal = tangentSpaceNormal * 2.0f - 1.0f;
        tangentSpaceNormal.x *= material.normalScale;
        tangentSpaceNormal.y *= material.normalScale;
        normalMapping(normal, glm::normalize(tangentSpaceNormal), intersection.tangent);
    }

    thrust::uniform_real_distribution<float> u01(0, 1);
    glm::vec3 newDir;
    Color color;

    if (u01(rng) < metallic) {
        const glm::vec3 reflect = glm::reflect(pathSegment.ray.direction, normal);
        newDir = material.useRoughSpecular
            ? calculateImperfectSpecularDirection(normal, reflect, intersection.tangent, rng, roughness)
            : reflect;
        color = metallic * albedo;
    }
    else {
        newDir = calculateRandomDirectionInHemisphere(normal, rng);
        color = (1.0f - metallic) * albedo;
    }

    pathSegment.ray.origin = intersect + newDir * 0.0001f;
    pathSegment.ray.direction = glm::normalize(newDir);
    pathSegment.color *= color;
}

__global__
void shadeBSDF(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    gpu::Material* materials,
    int materialCount,
    cudaTextureObject_t* textures,
    gpu::Environment env) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_paths) {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        PathSegment pathSegment = pathSegments[idx];

        if (intersection.t > 0.f) {
            if (intersection.materialId < 0 || intersection.materialId >= materialCount) {
                pathSegment.color = Color(0.0f);
                pathSegment.remainingBounces = 0;
                pathSegments[idx] = pathSegment;
                return;
            }

            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);

            const gpu::Material& material = materials[intersection.materialId];

            Color emissiveColor = material.emissiveColor * material.emissiveStrength;
            if (material.emissiveTexture.index >= 0) {
                emissiveColor *= sampleTexture(textures[material.emissiveTexture.index], intersection.uv);
            }

            // If the material indicates that the object was a light, "light" the ray.
            if (glm::length(emissiveColor) > 0.0f) {
                pathSegment.color *= emissiveColor;
                pathSegment.remainingBounces = 0;
            }
            else {
                scatterRay(pathSegment, intersection, material, textures, rng);
                --pathSegment.remainingBounces;
            }
        }
        else {
            if (env.enabled) {
                pathSegment.color *= sampleEnvironment(
                    textures[env.texture.index],
                    pathSegment.ray.direction,
                    env.intensity,
                    env.rotation);
            }
            else {
                pathSegment.color = Color(0.f);
            }
            pathSegment.remainingBounces = 0;
        }

        pathSegments[idx] = pathSegment;
    }
}

__global__
void generateGBuffer(
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    gpu::GBufferView gBuffer,
    gpu::Material* materials,
    int materialCount,
    cudaTextureObject_t* textures) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        const ShadeableIntersection& intersection = shadeableIntersections[idx];

        if (intersection.t <= 0.0f || intersection.materialId < 0 || intersection.materialId >= materialCount) {
            gBuffer.normal[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            gBuffer.albedo[idx] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            return;
        }

        gBuffer.normal[idx] = make_float4(
            intersection.surfaceNormal.x,
            intersection.surfaceNormal.y,
            intersection.surfaceNormal.z,
            0.0f);

        const gpu::Material& material = materials[intersection.materialId];
        Color albedo;
        if (material.baseColorTexture.index < 0) {
            albedo = material.baseColor;
        }
        else {
            albedo = sampleTexture(textures[material.baseColorTexture.index], intersection.uv);
        }
        gBuffer.albedo[idx] = make_float4(albedo.x, albedo.y, albedo.z, 1.0f);
    }
}

// Add the current iteration's output to the overall image.
__global__
void finalGather(int nPaths, float4* image, PathSegment* iterationPaths) {
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths) {
        PathSegment iterationPath = iterationPaths[index];
        float4& pixel = image[iterationPath.pixelIndex];
        pixel.x += iterationPath.color.x;
        pixel.y += iterationPath.color.y;
        pixel.z += iterationPath.color.z;
        pixel.w = 1.0f;
    }
}

__global__
void resolveAccumulationToColor(int pixelCount, const float4* accumulation, float4* color, int iter) {
    const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index >= pixelCount) {
        return;
    }

    const float invIter = 1.0f / static_cast<float>(iter);
    const float4 pixel = accumulation[index];
    color[index] = make_float4(pixel.x * invIter, pixel.y * invIter, pixel.z * invIter, 1.0f);
}

__global__
void stageOptixDenoiserGuides(
    int width,
    int height,
    const gpu::GBufferView gBuffer,
    float4* optixAlbedo,
    float4* optixNormal) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < width && y < height) {
        int index = x + (y * width);

        const float4 srcAlbedo = gBuffer.albedo[index];
        glm::vec3 albedo = glm::clamp(
            glm::vec3(srcAlbedo.x, srcAlbedo.y, srcAlbedo.z),
            glm::vec3(0.0f),
            glm::vec3(1.0f));
        optixAlbedo[index] = make_float4(albedo.x, albedo.y, albedo.z, 1.0f);

        const float4 srcNormal = gBuffer.normal[index];
        glm::vec3 n(srcNormal.x, srcNormal.y, srcNormal.z);
        if (glm::length2(n) == 0.0f) {
            optixNormal[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        else {
            n = glm::normalize(n);
            optixNormal[index] = make_float4(n.x, n.y, n.z, 0.0f);
        }
    }
}


struct PathTracer::Impl {
    void initialize();
    void setScene(const RenderScene& scene);
    void setCamera(const RenderCamera& camera);
    void setTraceDepth(int depth);
    void resize(int width, int height);
    void reset();
    void renderSample();
    void renderSample(bool denoise);
    void copyImageToPbo(uchar4* pbo, bool denoise);
    void copyGBufferToPbo(uchar4* pbo, int mode);
    void copyImageToHost(std::vector<glm::vec3>& out);
    void copyImageToHost(std::vector<glm::vec3>& out, bool useDenoised);
    bool hasDenoisedImage() const;
    void copyImageToPbo(uchar4* pbo);
    void shutdown();

private:
    void resizeTraceBuffers(size_t pixelCount);
    void initializeStreams();
    void destroyStreams();
    void initializeDenoiser();
    void destroyDenoiser();

    gpu::Scene gpuScene;
    gpu::Framebuffer frameBuffer;
    RenderSettings renderSettings;
    Camera camera = {};
    int traceDepth = 0;
    size_t pixelCount = 0;

    CudaBuffer<PathSegment> paths;
    CudaBuffer<ShadeableIntersection> intersections;
    CudaBuffer<ShadeableIntersection> firstIntersections;
    CudaBuffer<Hit> hits;
    PathSegment* currentPaths = nullptr;
    PathSegment* pathStart = nullptr;

    OptiXDenoiser* optixDenoiser = nullptr;
    OptixDeviceContext optixContext = nullptr;
    cudaStream_t stream1 = nullptr;
    cudaStream_t stream2 = nullptr;

#if TIMING
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
#endif

    int iteration = 0;
    bool denoisedAvailable = false;
};

void PathTracer::Impl::initialize() {
    initializeStreams();
}

void PathTracer::Impl::setScene(const RenderScene& scene) {
    gpuScene.upload(scene);
    renderSettings = scene.settings;
    iteration = 0;
    denoisedAvailable = false;
}

void PathTracer::Impl::setCamera(const RenderCamera& camera) {
    this->camera = makeKernelCamera(camera);
}

void PathTracer::Impl::setTraceDepth(int depth) {
    traceDepth = depth;
}

void PathTracer::Impl::resize(int width, int height) {
    frameBuffer.resize(width, height);
    resizeTraceBuffers(static_cast<size_t>(width) * static_cast<size_t>(height));
    iteration = 0;
    denoisedAvailable = false;
}

void PathTracer::Impl::resizeTraceBuffers(size_t pixelCount) {
    this->pixelCount = pixelCount;
    paths.allocate(pixelCount);
    intersections.allocate(pixelCount);
    hits.allocate(pixelCount);
#if CACHE_FIRST_BOUNCE
    firstIntersections.allocate(pixelCount);
#endif

    pathStart = paths.data();
    currentPaths = pathStart;

    if (!intersections.empty()) {
        cudaMemset(intersections.data(), 0, intersections.size() * sizeof(ShadeableIntersection));
    }
    if (!hits.empty()) {
        cudaMemset(hits.data(), 0, hits.size() * sizeof(Hit));
    }
#if CACHE_FIRST_BOUNCE
    if (!firstIntersections.empty()) {
        cudaMemset(firstIntersections.data(), 0, firstIntersections.size() * sizeof(ShadeableIntersection));
    }
#endif
}

void PathTracer::Impl::reset() {
    iteration = 0;
    denoisedAvailable = false;
    currentPaths = pathStart;
    frameBuffer.clear();
}

void PathTracer::Impl::renderSample() {
    renderSample(false);
}

void PathTracer::Impl::renderSample(bool denoise) {
    initializeStreams();
    if (denoise && !optixDenoiser) {
        initializeDenoiser();
    }
    if (!denoise) {
        denoisedAvailable = false;
    }

    const int iter = iteration + 1;
    const int pixelcount = static_cast<int>(pixelCount);
    gpu::SceneView scene = gpuScene.view();

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (camera.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (camera.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const size_t blockSize1d = 128;

    currentPaths = pathStart;

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d, 0, stream1>>>(
        camera,
        iter,
        traceDepth,
        currentPaths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = pathStart + pixelcount;
    int N = pixelcount;
    int num_paths = N;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    ShadeableIntersection* activeIntersections = NULL;

    while (!iterationComplete) {
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
        if (depth == 0 && iter != 1) {
            activeIntersections = firstIntersections.data();
        }
#endif

        if (activeIntersections == NULL) {
            traceClosestHits<<<numblocksPathSegmentTracing, blockSize1d, 0, stream1>>>(
                num_paths,
                scene,
                currentPaths,
                hits.data());
            resolveHitData<<<numblocksPathSegmentTracing, blockSize1d, 0, stream1>>>(
                num_paths,
                scene,
                currentPaths,
                hits.data(),
                intersections.data());
            checkCUDAError("Compute Intersections");

#if CACHE_FIRST_BOUNCE
            // NOTE: Copy before sorting since dev_first_intersections should map to unsorted dev_paths.
            if (depth == 0 && iter == 1) {
                cudaMemcpyAsync(
                    firstIntersections.data(),
                    intersections.data(),
                    pixelcount * sizeof(ShadeableIntersection),
                    cudaMemcpyDeviceToDevice,
                    stream2);
            }
#endif

#if RAY_SORTING
            // Sort the intersections and rays based on material types.
            thrust::sort_by_key(
                thrust::device,
                intersections.data(),
                intersections.data() + num_paths,
                currentPaths,
                compareIntersections());
#endif

            activeIntersections = intersections.data();
        }

        if (denoise && depth == 0) {
            generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d, 0, stream2>>>(
                num_paths,
                activeIntersections,
                frameBuffer.gbuffer.view(),
                scene.materials,
                scene.materialCount,
                scene.textureObjects);
        }

#if TIMING
        cudaEventRecord(startEvent);
#endif

        shadeBSDF<<<numblocksPathSegmentTracing, blockSize1d, 0, stream1>>>(
            iter,
            num_paths,
            activeIntersections,
            currentPaths,
            scene.materials,
            scene.materialCount,
            scene.textureObjects,
            scene.environment);
        checkCUDAError("shadeBSDF failed");

        // Partition the buffer based on whether the ray path is completed.
        currentPaths = thrust::partition(
            thrust::device,
            currentPaths,
            currentPaths + num_paths,
            isPathCompleted());

        num_paths = dev_path_end - currentPaths;

        iterationComplete = num_paths == 0;

        activeIntersections = NULL;

#if TIMING
        cudaEventRecord(endEvent);
        cudaEventSynchronize(endEvent);
        float ms;
        cudaEventElapsedTime(&ms, startEvent, endEvent);
#endif

        ++depth;
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d, 0, stream1>>>(
        N,
        frameBuffer.accumulation.data(),
        pathStart);
    checkCUDAError("Final Gather");

    resolveAccumulationToColor<<<numBlocksPixels, blockSize1d, 0, stream1>>>(
        pixelcount,
        frameBuffer.accumulation.data(),
        frameBuffer.color.data(),
        iter);
    checkCUDAError("Resolve accumulation to color");
    
    // Reset pointers to the start
    currentPaths = pathStart;

    if (denoise) {
        const gpu::GBufferView gbuffer = frameBuffer.gbuffer.view();
        const OptixImage2D optixNoisyColor = frameBuffer.colorInputImage();
        const OptixImage2D optixDenoisedColor = frameBuffer.denoisedImage();
        const OptixImage2D optixAlbedoGuide = frameBuffer.albedoImage();
        const OptixImage2D optixNormalGuide = frameBuffer.normalImage();

        stageOptixDenoiserGuides<<<blocksPerGrid2d, blockSize2d, 0, stream1>>>(
            camera.resolution.x,
            camera.resolution.y,
            gbuffer,
            gbuffer.albedo,
            gbuffer.normal);
        checkCUDAError("Prepare OptiX denoiser inputs");

        OptiXDenoiser::InputData inputData = {};
        inputData.color = optixNoisyColor;
        inputData.albedo = optixAlbedoGuide;
        inputData.normal = optixNormalGuide;

        OptiXDenoiser::OutputData outputData = {};
        outputData.color = optixDenoisedColor;

        if (!optixDenoiser->denoise(outputData, inputData, stream1)) {
            std::cerr << "OptiX denoiser invocation failed" << std::endl;
            exit(EXIT_FAILURE);
        }

        checkCUDAError("Run OptiX denoiser");
        denoisedAvailable = true;
    }

    iteration = iter;
}

void PathTracer::Impl::copyGBufferToPbo(uchar4* pbo, int mode) {
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (camera.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (camera.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    gbufferToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo,
        camera.resolution,
        frameBuffer.gbuffer.view(),
        mode);
    checkCUDAError("send gbuffer to PBO");
}

void PathTracer::Impl::copyImageToPbo(uchar4* pbo, bool denoise) {
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (camera.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (camera.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering.
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(
        pbo,
        camera.resolution,
        denoise ? frameBuffer.denoised.data() : frameBuffer.color.data());
    checkCUDAError("send image to PBO");
}

void PathTracer::Impl::copyImageToHost(std::vector<glm::vec3>& out) {
    frameBuffer.copyToHost(out);
}

void PathTracer::Impl::copyImageToHost(std::vector<glm::vec3>& out, bool useDenoised) {
    frameBuffer.copyToHost(out, useDenoised);
}

bool PathTracer::Impl::hasDenoisedImage() const {
    return denoisedAvailable;
}

void PathTracer::Impl::copyImageToPbo(uchar4* pbo) {
    copyImageToPbo(pbo, false);
}

void PathTracer::Impl::initializeStreams() {
#if TIMING
    if (!startEvent) {
        cudaEventCreate(&startEvent);
    }
    if (!endEvent) {
        cudaEventCreate(&endEvent);
    }
#endif

#if CUDA_STREAM
    if (!stream1) {
        cudaStreamCreate(&stream1);
    }
    if (!stream2) {
        cudaStreamCreate(&stream2);
    }
#else
    stream1 = 0;
    stream2 = 0;
#endif
}

void PathTracer::Impl::destroyStreams() {
#if TIMING
    if (startEvent) {
        cudaEventDestroy(startEvent);
        startEvent = nullptr;
    }
    if (endEvent) {
        cudaEventDestroy(endEvent);
        endEvent = nullptr;
    }
#endif

#if CUDA_STREAM
    if (stream1) {
        cudaStreamDestroy(stream1);
    }
    if (stream2) {
        cudaStreamDestroy(stream2);
    }
    stream1 = nullptr;
    stream2 = nullptr;
#endif
}

void PathTracer::Impl::initializeDenoiser() {
    initializeStreams();

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixDenoiserLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));

    optixDenoiser = new OptiXDenoiser(optixDenoiserLogCallback, nullptr);
    if (!optixDenoiser->init(
            optixContext,
            stream1,
            camera.resolution.x,
            camera.resolution.y,
            0,
            0,
            false,
            true,
            true,
            false,
            false)) {
        std::cerr << "Failed to initialize OptiX denoiser" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void PathTracer::Impl::destroyDenoiser() {
    if (optixDenoiser) {
        optixDenoiser->exit();
        delete optixDenoiser;
        optixDenoiser = nullptr;
    }

    if (optixContext) {
        OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
        optixContext = nullptr;
    }
}

void PathTracer::Impl::shutdown() {
    destroyDenoiser();
    destroyStreams();
    gpuScene.reset();
    frameBuffer.reset();
    paths.reset();
    intersections.reset();
    firstIntersections.reset();
    hits.reset();
    currentPaths = nullptr;
    pathStart = nullptr;
    pixelCount = 0;
    iteration = 0;
}

PathTracer::PathTracer()
    : impl(std::make_unique<Impl>()) {}

PathTracer::~PathTracer() = default;

PathTracer::PathTracer(PathTracer&&) noexcept = default;
PathTracer& PathTracer::operator=(PathTracer&&) noexcept = default;

void PathTracer::initialize() { impl->initialize(); }
void PathTracer::setScene(const RenderScene& scene) { impl->setScene(scene); }
void PathTracer::setCamera(const RenderCamera& camera) { impl->setCamera(camera); }
void PathTracer::setTraceDepth(int depth) { impl->setTraceDepth(depth); }
void PathTracer::resize(int width, int height) { impl->resize(width, height); }
void PathTracer::reset() { impl->reset(); }
void PathTracer::renderSample() { impl->renderSample(); }
void PathTracer::renderSample(bool denoise) { impl->renderSample(denoise); }
void PathTracer::copyImageToPbo(uchar4* pbo, bool denoise) {
    impl->copyImageToPbo(pbo, denoise);
}
void PathTracer::copyGBufferToPbo(uchar4* pbo, int mode) { impl->copyGBufferToPbo(pbo, mode); }
void PathTracer::copyImageToHost(std::vector<glm::vec3>& out) { impl->copyImageToHost(out); }
void PathTracer::copyImageToHost(std::vector<glm::vec3>& out, bool useDenoised) {
    impl->copyImageToHost(out, useDenoised);
}
bool PathTracer::hasDenoisedImage() const { return impl->hasDenoisedImage(); }
void PathTracer::copyImageToPbo(uchar4* pbo) { impl->copyImageToPbo(pbo); }
void PathTracer::shutdown() { impl->shutdown(); }
