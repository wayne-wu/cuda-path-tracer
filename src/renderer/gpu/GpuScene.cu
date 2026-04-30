#include "GpuScene.h"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

struct GpuSceneData {
    std::vector<gpu::Mesh> meshes;
    std::vector<gpu::Instance> instances;
    std::vector<gpu::Primitive> primitives;
    std::vector<gpu::Triangle> triangles;
    std::vector<gpu::Material> materials;

    std::vector<gpu::Vec3> vertices;
    std::vector<gpu::Vec3> normals;
    std::vector<uint32_t> indices;
    std::vector<gpu::Vec2> uvs;
    std::vector<gpu::Vec4> tangents;
    std::vector<gpu::Bin> bins;
    std::vector<int> binFaces;
};

gpu::TextureRef toGpuTextureRef(const RenderTextureRef& ref) {
    gpu::TextureRef gpuRef;
    gpuRef.index = ref.textureId;
    gpuRef.texCoord = ref.texCoord;
    return gpuRef;
}

gpu::GeometryType toGpuGeometryType(RenderGeometryType type) {
    switch (type) {
    case RenderGeometryType::Sphere:
        return gpu::GeometryType::Sphere;
    case RenderGeometryType::Cube:
        return gpu::GeometryType::Cube;
    case RenderGeometryType::Mesh:
    default:
        return gpu::GeometryType::Mesh;
    }
}

gpu::Material toGpuMaterial(const RenderMaterial& material) {
    gpu::Material gpuMaterial;
    gpuMaterial.baseColor = material.baseColor;
    gpuMaterial.baseColorTexture = toGpuTextureRef(material.baseColorTexture);
    gpuMaterial.metallic = material.metallic;
    gpuMaterial.roughness = material.roughness;
    gpuMaterial.metallicRoughnessTexture = toGpuTextureRef(material.metallicRoughnessTexture);
    gpuMaterial.emissiveColor = material.emissiveColor;
    gpuMaterial.emissiveStrength = material.emissiveStrength;
    gpuMaterial.emissiveTexture = toGpuTextureRef(material.emissiveTexture);
    gpuMaterial.normalTexture = toGpuTextureRef(material.normalTexture);
    gpuMaterial.normalScale = material.normalScale;
    gpuMaterial.transmission = material.transmission;
    gpuMaterial.ior = material.ior;
    gpuMaterial.alpha = material.alpha;
    gpuMaterial.doubleSided = material.doubleSided;
    gpuMaterial.useRoughSpecular = material.useRoughSpecular;
    return gpuMaterial;
}

gpu::Instance toGpuInstance(const RenderInstance& instance) {
    gpu::Instance gpuInstance;
    gpuInstance.geometryType = toGpuGeometryType(instance.geometryType);
    gpuInstance.meshId = instance.meshId;
    gpuInstance.materialId = instance.materialId;
    gpuInstance.transform = instance.transform;
    gpuInstance.inverseTransform = instance.inverseTransform;
    gpuInstance.inverseTranspose = instance.inverseTranspose;
    gpuInstance.bboxMin = instance.bboxMin;
    gpuInstance.bboxMax = instance.bboxMax;
    return gpuInstance;
}

gpu::Environment toGpuEnvironment(const RenderEnvironment& environment, int textureIndex) {
    gpu::Environment gpuEnvironment;
    gpuEnvironment.enabled = environment.enabled;
    gpuEnvironment.intensity = environment.intensity;
    gpuEnvironment.rotation = environment.rotation;
    gpuEnvironment.texture.index = textureIndex;
    return gpuEnvironment;
}

gpu::Bin toGpuBin(const RenderBin& bin) {
    gpu::Bin gpuBin;
    gpuBin.bbox_min = bin.bboxMin;
    gpuBin.bbox_max = bin.bboxMax;
    gpuBin.childIndex = bin.childIndex;
    gpuBin.startIndex = bin.startIndex;
    gpuBin.endIndex = bin.endIndex;
    gpuBin.depth = bin.depth;
    gpuBin.padding = gpu::Vec2(0.0f);
    return gpuBin;
}

void appendTrianglesForMesh(GpuSceneData& data, const RenderMesh& mesh) {
    const size_t triangleIndexCount = mesh.indices.size() - (mesh.indices.size() % 3);
    data.triangles.reserve(data.triangles.size() + triangleIndexCount / 3);

    for (size_t i = 0; i < triangleIndexCount; i += 3) {
        const uint32_t i0 = mesh.indices[i + 0];
        const uint32_t i1 = mesh.indices[i + 1];
        const uint32_t i2 = mesh.indices[i + 2];

        if (i0 >= mesh.positions.size() || i1 >= mesh.positions.size() || i2 >= mesh.positions.size()) {
            continue;
        }

        gpu::Triangle triangle;
        triangle.v0 = mesh.positions[i0];
        triangle.v1 = mesh.positions[i1];
        triangle.v2 = mesh.positions[i2];
        data.triangles.push_back(triangle);
    }
}

void appendMesh(GpuSceneData& data, const RenderMesh& mesh) {
    gpu::Primitive primitive;
    primitive.count = static_cast<int>(mesh.indices.size());
    primitive.num_vertices = static_cast<int>(mesh.positions.size());
    primitive.i_offset = static_cast<gpu::Offset>(data.indices.size());
    primitive.v_offset = static_cast<gpu::Offset>(data.vertices.size());
    primitive.face_offset = static_cast<gpu::Offset>(data.triangles.size());
    primitive.bin_offset = mesh.bins.empty() ? -1 : static_cast<gpu::Offset>(data.bins.size());
    primitive.bf_offset = mesh.binFaces.empty() ? -1 : static_cast<gpu::Offset>(data.binFaces.size());
    primitive.n_offset = mesh.normals.empty() ? -1 : static_cast<gpu::Offset>(data.normals.size());
    primitive.uv_offset = mesh.uvs.empty() ? -1 : static_cast<gpu::Offset>(data.uvs.size());
    primitive.t_offset = mesh.tangents.empty() ? -1 : static_cast<gpu::Offset>(data.tangents.size());

    if (!mesh.positions.empty()) {
        primitive.bbox_min = mesh.positions.front();
        primitive.bbox_max = mesh.positions.front();
        for (const gpu::Vec3& position : mesh.positions) {
            primitive.bbox_min = glm::min(primitive.bbox_min, position);
            primitive.bbox_max = glm::max(primitive.bbox_max, position);
        }
    }

    gpu::Mesh gpuMesh;
    gpuMesh.prim_count = 1;
    gpuMesh.prim_offset = static_cast<int>(data.primitives.size());

    data.meshes.push_back(gpuMesh);
    data.primitives.push_back(primitive);

    data.vertices.insert(data.vertices.end(), mesh.positions.begin(), mesh.positions.end());
    data.normals.insert(data.normals.end(), mesh.normals.begin(), mesh.normals.end());
    data.uvs.insert(data.uvs.end(), mesh.uvs.begin(), mesh.uvs.end());
    data.tangents.insert(data.tangents.end(), mesh.tangents.begin(), mesh.tangents.end());
    data.indices.insert(data.indices.end(), mesh.indices.begin(), mesh.indices.end());
    data.binFaces.insert(data.binFaces.end(), mesh.binFaces.begin(), mesh.binFaces.end());
    data.bins.reserve(data.bins.size() + mesh.bins.size());
    for (const RenderBin& bin : mesh.bins) {
        data.bins.push_back(toGpuBin(bin));
    }

    appendTrianglesForMesh(data, mesh);
}

GpuSceneData buildGpuSceneData(const RenderScene& scene) {
    GpuSceneData data;

    data.materials.reserve(scene.materials.size());
    for (const RenderMaterial& material : scene.materials) {
        data.materials.push_back(toGpuMaterial(material));
    }

    data.meshes.reserve(scene.meshes.size());
    data.primitives.reserve(scene.meshes.size());
    for (const RenderMesh& mesh : scene.meshes) {
        appendMesh(data, mesh);
    }

    data.instances.reserve(scene.instances.size());
    for (const RenderInstance& instance : scene.instances) {
        data.instances.push_back(toGpuInstance(instance));
    }

    return data;
}

std::vector<unsigned char> toRgba8(const RenderTexture& texture) {
    std::vector<unsigned char> rgba;
    if (texture.width <= 0 || texture.height <= 0 || texture.pixels.empty()) {
        return rgba;
    }

    const int components = std::max(texture.components, 1);
    const size_t pixelCount = static_cast<size_t>(texture.width) * static_cast<size_t>(texture.height);
    rgba.resize(pixelCount * 4);

    for (size_t i = 0; i < pixelCount; ++i) {
        const size_t src = i * components;
        const size_t dst = i * 4;
        rgba[dst + 0] = texture.pixels[src + 0];
        rgba[dst + 1] = components > 1 ? texture.pixels[src + 1] : texture.pixels[src + 0];
        rgba[dst + 2] = components > 2 ? texture.pixels[src + 2] : texture.pixels[src + 0];
        rgba[dst + 3] = components > 3 ? texture.pixels[src + 3] : 255;
    }

    return rgba;
}

} // namespace

void gpu::Scene::upload(const RenderScene& scene) {
    GpuSceneData staging = buildGpuSceneData(scene);

    meshes.upload(staging.meshes);
    instances.upload(staging.instances);
    primitives.upload(staging.primitives);
    triangles.upload(staging.triangles);
    materials.upload(staging.materials);
    vertices.upload(staging.vertices);
    normals.upload(staging.normals);
    indices.upload(staging.indices);
    uvs.upload(staging.uvs);
    tangents.upload(staging.tangents);
    bins.upload(staging.bins);
    binFaces.upload(staging.binFaces);

    uploadTextures(scene);
    environment = toGpuEnvironment(scene.environment, scene.environment.enabled ? static_cast<int>(textures.size()) - 1 : -1);
}

void gpu::Scene::reset() {
    meshes.reset();
    instances.reset();
    primitives.reset();
    triangles.reset();
    materials.reset();
    vertices.reset();
    normals.reset();
    indices.reset();
    uvs.reset();
    tangents.reset();
    bins.reset();
    binFaces.reset();
    textureObjects.reset();
    textures.clear();
    environment = Environment{};
}

gpu::SceneView gpu::Scene::view() {
    SceneView sceneView;
    sceneView.meshes = meshes.data();
    sceneView.instances = instances.data();
    sceneView.materials = materials.data();
    sceneView.textureObjects = textureObjects.data();
    sceneView.primData.primitives = primitives.data();
    sceneView.primData.triangles = triangles.data();
    sceneView.primData.vertices = vertices.data();
    sceneView.primData.normals = normals.data();
    sceneView.primData.indices = indices.data();
    sceneView.primData.uvs = uvs.data();
    sceneView.primData.tangents = tangents.data();
    sceneView.primData.bins = bins.data();
    sceneView.primData.binFaces = binFaces.data();
    sceneView.environment = environment;
    sceneView.meshCount = static_cast<int>(meshes.size());
    sceneView.instanceCount = static_cast<int>(instances.size());
    sceneView.materialCount = static_cast<int>(materials.size());
    sceneView.textureCount = static_cast<int>(textures.size());
    return sceneView;
}

void gpu::Scene::uploadTextures(const RenderScene& scene) {
    textures.clear();
    textures.reserve(scene.textures.size() + (scene.environment.enabled ? 1 : 0));

    std::vector<cudaTextureObject_t> textureHandles;
    textureHandles.reserve(scene.textures.size() + (scene.environment.enabled ? 1 : 0));

    for (const RenderTexture& texture : scene.textures) {
        textures.emplace_back();
        std::vector<unsigned char> rgba = toRgba8(texture);
        if (!rgba.empty()) {
            textures.back().uploadRGBA8(texture.width, texture.height, rgba.data());
        }
        textureHandles.push_back(textures.back().object());
    }

    if (scene.environment.enabled && scene.environment.width > 0 && scene.environment.height > 0 && !scene.environment.pixels.empty()) {
        textures.emplace_back();
        textures.back().uploadRGBA32F(
            scene.environment.width,
            scene.environment.height,
            reinterpret_cast<const float*>(scene.environment.pixels.data()));
        textureHandles.push_back(textures.back().object());
    }

    textureObjects.upload(textureHandles);
}
