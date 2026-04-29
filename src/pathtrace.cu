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

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

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

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
        int iter, glm::vec3* image) {
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y) {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int) (pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int) (pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int) (pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

__global__ void gbufferToPBO(uchar4* pbo, glm::ivec2 resolution, GBufferPixel* gBuffer, int mode) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < resolution.x && y < resolution.y) {
    int index = x + (y * resolution.x);

    switch (mode) {
    case GBUFFER_TIME:
    case GBUFFER_POSITION: {
      glm::vec3 albedo = glm::clamp(gBuffer[index].albedo, glm::vec3(0.0f), glm::vec3(1.0f));
      pbo[index].w = 0;
      pbo[index].x = static_cast<unsigned char>(albedo.x * 255.0f);
      pbo[index].y = static_cast<unsigned char>(albedo.y * 255.0f);
      pbo[index].z = static_cast<unsigned char>(albedo.z * 255.0f);
      break;
    }

    case GBUFFER_NORMAL:
      glm::vec3 n = gBuffer[index].n;
      pbo[index].w = 0;
      pbo[index].x = abs((int)(n.x * 255.0));
      pbo[index].y = abs((int)(n.y * 255.0));
      pbo[index].z = abs((int)(n.z * 255.0));
      break;
    }
  }
}

//Static variables for device memory, any extra info you need, etc
static Scene * hst_scene = NULL;
static glm::vec3 * dev_image = NULL;
static Geom * dev_geoms = NULL;
static Mesh * dev_meshes = NULL;
static Material * dev_materials = NULL;
static PathSegment * dev_paths = NULL;  
static PathSegment * dev_paths_start = NULL;
static ShadeableIntersection * dev_intersections = NULL;
static ShadeableIntersection * dev_first_intersections = NULL;
static Hit * dev_hits = NULL;

// Texture Data
static cudaTextureObject_t * dev_texObjs = NULL;
static std::vector<cudaArray_t> dev_texArrays;
static std::vector<cudaTextureObject_t> texObjs;
static cudaArray_t dev_envArray = NULL;
static cudaTextureObject_t dev_envTexObj = 0;

// Mesh Data for the GPU
static PrimData dev_prim_data;

// Denoising
static glm::vec3* dev_denoised_image = NULL;
static GBufferPixel* dev_gBuffer = NULL;
static OptiXDenoiser* optixDenoiser = NULL;
static OptixDeviceContext optixContext = nullptr;
static OptixImage2D optixNoisyColor = {};
static OptixImage2D optixDenoisedColor = {};
static OptixImage2D optixAlbedoGuide = {};
static OptixImage2D optixNormalGuide = {};

#if TIMING
static cudaEvent_t startEvent = NULL;
static cudaEvent_t endEvent = NULL;
#endif

// Streams
static cudaStream_t stream1 = NULL;
static cudaStream_t stream2 = NULL;

static void optixDenoiserLogCallback(unsigned int level, const char* tag, const char* message, void*) {
    std::cerr << "[OptiX][" << level << "][" << tag << "] " << message << std::endl;
}

static void destroyOptixDenoiser() {
    if (optixDenoiser) {
        optixDenoiser->exit();
        delete optixDenoiser;
        optixDenoiser = NULL;
    }

    freeOptixImage2D(optixNoisyColor);
    freeOptixImage2D(optixDenoisedColor);
    freeOptixImage2D(optixAlbedoGuide);
    freeOptixImage2D(optixNormalGuide);

    if (optixContext) {
        OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
        optixContext = nullptr;
    }
}

static void initOptixDenoiser(const Camera& cam, cudaStream_t stream) {
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = &optixDenoiserLogCallback;
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0;  // Use the current CUDA context.
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &optixContext));

    optixDenoiser = new OptiXDenoiser(optixDenoiserLogCallback, nullptr);
    if (!optixDenoiser->init(
            optixContext,
            stream,
            cam.resolution.x,
            cam.resolution.y,
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

    optixNoisyColor = createOptixImage2D(cam.resolution.x, cam.resolution.y, OPTIX_PIXEL_FORMAT_FLOAT4);
    optixDenoisedColor = createOptixImage2D(cam.resolution.x, cam.resolution.y, OPTIX_PIXEL_FORMAT_FLOAT4);
    optixAlbedoGuide = createOptixImage2D(cam.resolution.x, cam.resolution.y, OPTIX_PIXEL_FORMAT_FLOAT4);
    optixNormalGuide = createOptixImage2D(cam.resolution.x, cam.resolution.y, OPTIX_PIXEL_FORMAT_FLOAT4);
}


template <class T>
void mallocAndCopy(T* &d, std::vector<T> &h) {
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
}

/**
* Initialize texture objects
* Based on: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-object-api
*/
void textureInit(const Texture& tex, int i) {
    // Allocate CUDA array in device memory
    cudaTextureObject_t texObj;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaMallocArray(&dev_texArrays[i], &channelDesc, tex.width, tex.height);

    // Set pitch of the source (the width in memory in bytes of the 2D array pointed
    // to by src, including padding), we dont have any padding
    // const size_t spitch = tex.width * sizeof(unsigned char);
    // Copy texture image in host memory to device memory
    cudaMemcpy2DToArray(dev_texArrays[i], 0, 0, tex.image, tex.width * tex.components * sizeof(unsigned char), tex.width * tex.components * sizeof(unsigned char), tex.height, cudaMemcpyHostToDevice);
    
    // Specify texture
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_texArrays[i];

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeNormalizedFloat;
    texDesc.normalizedCoords = 1;

    // Create texture object
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
    cudaMemcpy(dev_texObjs+i, &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
    checkCUDAError("textureInit failed");

    texObjs.push_back(texObj);
}

void environmentTextureInit(const EnvironmentMap& env) {
    if (!env.enabled || env.pixels.empty()) {
        return;
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    cudaMallocArray(&dev_envArray, &channelDesc, env.width, env.height);
    cudaMemcpy2DToArray(
        dev_envArray,
        0,
        0,
        env.pixels.data(),
        env.width * sizeof(glm::vec4),
        env.width * sizeof(glm::vec4),
        env.height,
        cudaMemcpyHostToDevice);

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = dev_envArray;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    cudaCreateTextureObject(&dev_envTexObj, &resDesc, &texDesc, NULL);
    checkCUDAError("environmentTextureInit failed");
}

void pathtraceInit(Scene *scene) {
    hst_scene = scene;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
    dev_paths_start = dev_paths;

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_hits, pixelcount * sizeof(Hit));
    cudaMemset(dev_hits, 0, pixelcount * sizeof(Hit));

    mallocAndCopy<Geom>(dev_geoms, scene->geoms);
    mallocAndCopy<Material>(dev_materials, scene->materials);
    mallocAndCopy<Mesh>(dev_meshes, scene->meshes);

    // Mesh GPU data malloc
    mallocAndCopy<Primitive>(dev_prim_data.primitives, scene->primitives);
    mallocAndCopy<Triangle>(dev_prim_data.triangles, scene->mesh_faces);
    mallocAndCopy<uint16_t>(dev_prim_data.indices, scene->mesh_indices);
    mallocAndCopy<glm::vec3>(dev_prim_data.vertices, scene->mesh_vertices);
    mallocAndCopy<glm::vec3>(dev_prim_data.normals, scene->mesh_normals);
    mallocAndCopy<glm::vec2>(dev_prim_data.uvs, scene->mesh_uvs);
    mallocAndCopy<glm::vec4>(dev_prim_data.tangents, scene->mesh_tangents);

    // Create Texture Memory
    texObjs.clear(); dev_texArrays.clear();
    cudaMalloc(&dev_texObjs, scene->textures.size()*sizeof(cudaTextureObject_t));
    dev_texArrays.resize(scene->textures.size());
    for (int i = 0; i < scene->textures.size(); i++)
      textureInit(scene->textures[i], i);
    environmentTextureInit(scene->environment);

#if OCTREE
    mallocAndCopy<Bin>(dev_prim_data.bins, scene->bins);
    mallocAndCopy<int>(dev_prim_data.binFaces, scene->faceBins);
#endif

#if CACHE_FIRST_BOUNCE
    cudaMalloc(&dev_first_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_first_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
#endif

#if TIMING
    cudaEventCreate(&startEvent);
    cudaEventCreate(&endEvent);
#endif

#if CUDA_STREAM
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
#else
    stream1 = 0;
    stream2 = 0;
#endif

    // Denoising
    initOptixDenoiser(cam, stream1);
    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));
    cudaMalloc(&dev_denoised_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_denoised_image, 0, pixelcount * sizeof(glm::vec3));

    checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
    dev_paths = dev_paths_start;

    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_meshes);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_hits);
    cudaFree(dev_texObjs);
    if (dev_envTexObj) {
        cudaDestroyTextureObject(dev_envTexObj);
    }
    if (dev_envArray) {
        cudaFreeArray(dev_envArray);
    }

    cudaFree(dev_gBuffer);
    cudaFree(dev_denoised_image);
    destroyOptixDenoiser();

    dev_image = NULL;
    dev_paths = NULL;
    dev_paths_start = NULL;
    dev_geoms = NULL;
    dev_meshes = NULL;
    dev_materials = NULL;
    dev_intersections = NULL;
    dev_hits = NULL;
    dev_texObjs = NULL;
    dev_envTexObj = 0;
    dev_envArray = NULL;
    dev_gBuffer = NULL;
    dev_denoised_image = NULL;
    hst_scene = NULL;

    // Mesh GPU data free
    dev_prim_data.free();

    for (int i = 0; i < texObjs.size(); i++) {
      cudaDestroyTextureObject(texObjs[i]);
      cudaFreeArray(dev_texArrays[i]);
    }
    texObjs.clear();
    dev_texArrays.clear();

#if CACHE_FIRST_BOUNCE
    cudaFree(dev_first_intersections);
    dev_first_intersections = NULL;
#endif
    
#if TIMING
    if (startEvent != NULL)
      cudaEventDestroy(startEvent);
    if (endEvent != NULL)
      cudaEventDestroy(endEvent);
    startEvent = NULL;
    endEvent = NULL;
#endif

#if CUDA_STREAM
    if(stream1) cudaStreamDestroy(stream1);
    if(stream2) cudaStreamDestroy(stream2);
    stream1 = NULL;
    stream2 = NULL;
#endif

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
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
__global__ void traceClosestHits(
      int num_paths
    , int num_prims
    , int num_geoms
    , PathSegment * pathSegments
    , Geom * geoms
    , Mesh * meshes
    , PrimData mesh_data
    , Hit * hits
)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        const PathSegment& pathSegment = pathSegments[path_index];
        Hit hit;

        // naive parse through global geoms
        for (int i = 0; i < num_geoms; i++)
        {
            const Geom& geom = geoms[i];
            if (geom.type == CUBE)
            {
                const float t = boxIntersectionTest(geom, pathSegment.ray);
                if (t > 0.f) {
                  Vec3 pointWS;
                  Vec3 normalWS;
                  resolveBoxHitData(geom, pathSegment.ray, t, pointWS, normalWS);
                  const float worldT = glm::length(pathSegment.ray.origin - pointWS);
                  if (hit.t < 0.0f || worldT < hit.t) {
                    hit = Hit();
                    hit.geomId = i;
                    hit.t = worldT;
                  }
                }
            }
            else if (geom.type == SPHERE)
            {
                const float t = sphereIntersectionTest(geom, pathSegment.ray);
                if (t > 0.f) {
                  Vec3 pointWS;
                  Vec3 normalWS;
                  resolveSphereHitData(geom, pathSegment.ray, t, pointWS, normalWS);
                  const float worldT = glm::length(pathSegment.ray.origin - pointWS);
                  if (hit.t < 0.0f || worldT < hit.t) {
                    hit = Hit();
                    hit.geomId = i;
                    hit.t = worldT;
                  }
                }
            }
            else if (geom.type == MESH)
            {
              Hit candidateHit;
              if (meshIntersectionTest(geom, meshes[geom.meshid], mesh_data.primitives, mesh_data, pathSegment.ray, candidateHit)) {
                  if (hit.t < 0.0f || candidateHit.t < hit.t) {
                    hit = candidateHit;
                    hit.geomId = i;
                  }
              }
            }
        }

        hits[path_index] = hit;
    }
}

__global__ void resolveHitData(
      int num_paths
    , int num_prims
    , PathSegment * pathSegments
    , Geom * geoms
    , PrimData mesh_data
    , Hit * hits
    , ShadeableIntersection * intersections
)
{
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

    const Geom& geom = geoms[hit.geomId];
    Vec3 pointWS(0.0f);

    if (geom.type == MESH) {
      const Primitive& prim = mesh_data.primitives[hit.primId];
      computeFaceInfo(mesh_data, prim, hit.faceId, hit.bary,
        intersection.surfaceNormal, intersection.uv, intersection.tangent);

      intersection.surfaceNormal = glm::normalize(
        multiplyMV0(geom.invTranspose, intersection.surfaceNormal));
      intersection.tangent = glm::vec4(
        glm::normalize(multiplyMV0(geom.invTranspose, glm::vec3(intersection.tangent))),
        intersection.tangent.w);
      intersection.materialId = prim.mat_id;
      pointWS = getPointOnRay(ray, hit.t);
    }
    else if (geom.type == CUBE) {
      const float objectT = boxIntersectionTest(geom, ray);
      resolveBoxHitData(geom, ray, objectT, pointWS, intersection.surfaceNormal);
      const Vec3 tangentDir = glm::abs(intersection.surfaceNormal.x) > 0.9f
        ? Vec3(0.0f, 1.0f, 0.0f)
        : Vec3(1.0f, 0.0f, 0.0f);
      intersection.tangent = glm::vec4(glm::normalize(glm::cross(tangentDir, intersection.surfaceNormal)), 1.0f);
      intersection.materialId = geom.materialid;
    }
    else if (geom.type == SPHERE) {
      const float objectT = sphereIntersectionTest(geom, ray);
      resolveSphereHitData(geom, ray, objectT, pointWS, intersection.surfaceNormal);
      const Vec3 tangentDir = glm::abs(intersection.surfaceNormal.x) > 0.9f
        ? Vec3(0.0f, 1.0f, 0.0f)
        : Vec3(1.0f, 0.0f, 0.0f);
      intersection.tangent = glm::vec4(glm::normalize(glm::cross(tangentDir, intersection.surfaceNormal)), 1.0f);
      intersection.materialId = geom.materialid;
    }

    if (glm::dot(ray.direction, intersection.surfaceNormal) > 0.0f) {
      intersection.surfaceNormal = -intersection.surfaceNormal;
      intersection.tangent.w = -intersection.tangent.w;
    }

    intersection.t = hit.t;
    intersections[path_index] = intersection;
}

__device__ Color sampleEnvironment(
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

__global__ void shadeBSDF(
  int iter
  , int num_paths
  , int mat_size
  , ShadeableIntersection* shadeableIntersections
  , PathSegment* pathSegments
  , Material* materials
  , cudaTextureObject_t* textures
  , cudaTextureObject_t envTexture
  , bool envEnabled
  , float envIntensity
  , float envRotation) {

  extern __shared__ Material mats[];

  int tx = threadIdx.x;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (tx < mat_size)
    mats[tx] = materials[tx];

  __syncthreads();

  if (idx < num_paths) {

    ShadeableIntersection intersection = shadeableIntersections[idx];
    PathSegment pathSegment = pathSegments[idx];

    if (intersection.t > 0.f) {
      thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
      thrust::uniform_real_distribution<float> u01(0, 1);

      const Material& material = mats[intersection.materialId];

      Color emissiveColor = material.emissiveFactor;
      if (material.emissiveTexture.index >= 0) {
        emissiveColor *= sampleTexture(textures[material.emissiveTexture.index], intersection.uv);
      }

      // If the material indicates that the object was a light, "light" the ray
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
      if (envEnabled) {
        pathSegment.color *= sampleEnvironment(envTexture, pathSegment.ray.direction, envIntensity, envRotation);
      }
      else {
        pathSegment.color = Color(0.f);
      }
      pathSegment.remainingBounces = 0;
    }

    pathSegments[idx] = pathSegment;
  }
}

__global__ void generateGBuffer(
  int num_paths,
  ShadeableIntersection* shadeableIntersections,
  GBufferPixel* gBuffer,
  Material* materials,
  cudaTextureObject_t* textures) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_paths)
  {
    const ShadeableIntersection& intersection = shadeableIntersections[idx];

    if (intersection.t <= 0.0f) {
      gBuffer[idx].n = glm::vec3(0.0f);
      gBuffer[idx].albedo = glm::vec3(0.0f);
      return;
    }

    gBuffer[idx].n = glm::normalize(intersection.surfaceNormal);

    const Material& material = materials[intersection.materialId];
    int txId = material.tex_offset + material.pbrMetallicRoughness.baseColorTexture.index;
    if (txId < 0) {
      gBuffer[idx].albedo = material.pbrMetallicRoughness.baseColorFactor;
    } else {
      gBuffer[idx].albedo = sampleTexture(textures[txId], intersection.uv);
    }
  }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3 * image, PathSegment * iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

// Calculate the weight for gBuffer data
__device__ inline float calculateWeight(glm::vec3& a, glm::vec3& b, float phi) {
  glm::vec3 t = a - b;
  float dist2 = glm::dot(t, t);
  return min(exp(-(dist2) / (phi + 0.0001f)), 1.f);
}

// Calculate the weight for the guassian filter
__device__ inline float gaussianWeight(int x, int y, float s) {
  return (1.0f / (2 * PI * s * s)) * exp(-(x * x + y * y) / (2 * s * s));
}

__global__ void normalizeImage(int width, int height, glm::vec3* image, int iter) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < width && y < height) {
    int index = x + (y * width);
    glm::vec3 pix = image[index];

    pix.x /= iter;
    pix.y /= iter;
    pix.z /= iter;

    image[index] = pix;
  }
}

__global__ void stageOptixDenoiserInputs(
  int width,
  int height,
  const glm::vec3* image,
  int iter,
  const GBufferPixel* gBuffer,
  float4* optixColor,
  float4* optixAlbedo,
  float4* optixNormal) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < width && y < height) {
    int index = x + (y * width);

    glm::vec3 color = image[index] / static_cast<float>(iter);
    optixColor[index] = make_float4(color.x, color.y, color.z, 1.0f);

    glm::vec3 albedo = glm::clamp(gBuffer[index].albedo, glm::vec3(0.0f), glm::vec3(1.0f));
    optixAlbedo[index] = make_float4(albedo.x, albedo.y, albedo.z, 1.0f);

    glm::vec3 n = gBuffer[index].n;
    if (glm::length2(n) == 0.0f) {
      optixNormal[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    else {
      n = glm::normalize(n);
      optixNormal[index] = make_float4(n.x, n.y, n.z, 0.0f);
    }
  }
}

__global__ void optixOutputToDenoisedImage(int width, int height, const float4* optixImage, glm::vec3* image) {
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  if (x < width && y < height) {
    int index = x + (y * width);
    float4 pix = optixImage[index];
    image[index] = glm::vec3(pix.x, pix.y, pix.z);
  }
}

// NOT USED: Denoise Kernel
__global__ void kernDenoise(int width, int height, glm::vec3* image,
  int filterSize, GBufferPixel* gBuffer, int stepWidth, glm::mat4 camView, glm::mat4 camProj,
  float colorWeight, float normalWeight, float positionWeight) {

  // 5x5 B3-spline filter
  float kernel[5][5] = {
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625,
    0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
    0.0234375, 0.09375, 0.140625, 0.09375, 0.0234375,
    0.015625, 0.0625, 0.09375, 0.0625, 0.015625,
    0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625 };

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * width);

  if (index < width * height) {

    glm::vec3 c0 = image[index];
    glm::vec3 c1 = glm::vec3(0.f);
    glm::vec3 dSum = glm::vec3(0.f);

    float k = 0.f;
    for (int i = -2; i <= 2; i++) {
      for (int j = -2; j <= 2; j++) {
        int x0 = x + i * stepWidth;
        int y0 = y + j * stepWidth;
        // Check if the x and y are within bound
        if (x0 >= 0 && x0 < width && y0 >= 0 && y0 < height)
        {
          int idx = x0 + y0 * width;

          float weight = 1.f;

#if USE_GBUFFER
          float c_w = calculateWeight(image[index], image[idx], colorWeight);
          float n_w = calculateWeight(gBuffer[index].n, gBuffer[idx].n, normalWeight);
          float a_w = calculateWeight(gBuffer[index].albedo, gBuffer[idx].albedo, positionWeight);
          weight = c_w * n_w * a_w;
#endif

          float ker = kernel[i + 2][j + 2];
          c1 += weight * ker * image[idx];
          k += weight * ker;
        }
      }
    }

    image[index] = c1 / k;
  }
}


/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter, bool denoise) 
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera &cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const size_t blockSize1d = 128;

    generateRayFromCamera <<<blocksPerGrid2d, blockSize2d, 0, stream1 >>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int N = pixelcount;
    int num_paths = N;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    ShadeableIntersection* intersections = NULL;

    while (!iterationComplete) {

      dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;

#if CACHE_FIRST_BOUNCE
      if (depth == 0 && iter != 1)
        intersections = dev_first_intersections;
#endif

      if (intersections == NULL) {
        int num_prims = hst_scene->primitives.size();
        traceClosestHits<<<numblocksPathSegmentTracing, blockSize1d, 0, stream1>>>(
          num_paths
          , num_prims
          , hst_scene->geoms.size()
          , dev_paths
          , dev_geoms
          , dev_meshes
          , dev_prim_data
          , dev_hits);
        resolveHitData<<<numblocksPathSegmentTracing, blockSize1d, 0, stream1>>>(
          num_paths
          , num_prims
          , dev_paths
          , dev_geoms
          , dev_prim_data
          , dev_hits
          , dev_intersections);
        checkCUDAError("Compute Intersections");
        
#if CACHE_FIRST_BOUNCE
        // NOTE: Copy before sorting since dev_first_intersections should map to unsorted dev_paths
        if(depth == 0 && iter == 1)
          cudaMemcpyAsync(dev_first_intersections, dev_intersections, 
            pixelcount * sizeof(ShadeableIntersection), cudaMemcpyDeviceToDevice, stream2);
#endif

#if RAY_SORTING
        // sort the intersections and rays based on material types
        thrust::sort_by_key(thrust::device, dev_intersections, dev_intersections + num_paths, dev_paths, compareIntersections());
#endif

        intersections = dev_intersections;
      }

      if (denoise && depth == 0) {
        generateGBuffer<<<numblocksPathSegmentTracing, blockSize1d>>>(
          num_paths,
          intersections,
          dev_gBuffer,
          dev_materials,
          dev_texObjs);
      }

#if TIMING
      cudaEventRecord(startEvent);
#endif

      int matSize = hst_scene->materials.size();
      shadeBSDF <<<numblocksPathSegmentTracing, blockSize1d, matSize*sizeof(Material), stream1>>> (
        iter,
        num_paths,
        matSize,
        intersections,
        dev_paths,
        dev_materials,
        dev_texObjs,
        dev_envTexObj,
        hst_scene->environment.enabled,
        hst_scene->environment.intensity,
        hst_scene->environment.rotation
        );
      checkCUDAError("shadeBSDF failed");

      // partition (stream compaction) the buffer based on whether the ray path is completed
      dev_paths = thrust::partition(thrust::device, dev_paths, dev_paths + num_paths, isPathCompleted());

      num_paths = dev_path_end - dev_paths;

      iterationComplete = num_paths == 0;

      intersections = NULL;

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
    finalGather<<<numBlocksPixels, blockSize1d, 0, stream1>>>(N, dev_image, dev_paths_start);
    checkCUDAError("Final Gather");
    
    // Reset pointers to the start
    dev_paths = dev_paths_start;

    if (denoise) {
      stageOptixDenoiserInputs<<<blocksPerGrid2d, blockSize2d, 0, stream1>>>(
        cam.resolution.x,
        cam.resolution.y,
        dev_image,
        iter,
        dev_gBuffer,
        reinterpret_cast<float4*>(optixNoisyColor.data),
        reinterpret_cast<float4*>(optixAlbedoGuide.data),
        reinterpret_cast<float4*>(optixNormalGuide.data));
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

      optixOutputToDenoisedImage<<<blocksPerGrid2d, blockSize2d, 0, stream1>>>(
        cam.resolution.x,
        cam.resolution.y,
        reinterpret_cast<const float4*>(optixDenoisedColor.data),
        dev_denoised_image);
      checkCUDAError("Read OptiX denoiser output");
    }
}


// CHECKITOUT: this kernel "post-processes" the gbuffer/gbuffers into something that you can visualize for debugging.
void showGBuffer(uchar4* pbo, int mode) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // CHECKITOUT: process the gbuffer results and send them to OpenGL buffer for visualization
  gbufferToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, dev_gBuffer, mode);
  checkCUDAError("send gbuffer to PBO");
}

void showImage(uchar4* pbo, int iter, bool denoise) {
  const Camera& cam = hst_scene->state.camera;
  const dim3 blockSize2d(8, 8);
  const dim3 blocksPerGrid2d(
    (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
    (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

  // Send results to OpenGL buffer for rendering
  sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, denoise? 1 : iter, denoise ? dev_denoised_image : dev_image);
  checkCUDAError("send image to PBO");
}
