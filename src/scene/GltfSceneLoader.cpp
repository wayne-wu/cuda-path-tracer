#include "GltfSceneLoader.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>

#define TINYGLTF_IMPLEMENTATION
#include "tiny_gltf.h"
#include "core/utilities.h"

namespace {

std::string getFilePathExtension(const std::string& filename) {
    const size_t dot = filename.find_last_of('.');
    if (dot == std::string::npos) {
        return "";
    }
    return filename.substr(dot + 1);
}

glm::mat4 getNodeTransform(const tinygltf::Node& node) {
    if (!node.matrix.empty()) {
        return glm::make_mat4(node.matrix.data());
    }

    const glm::vec3 translation = node.translation.empty()
        ? glm::vec3(0.0f)
        : glm::make_vec3(node.translation.data());
    const glm::quat rotation = node.rotation.empty()
        ? glm::quat(1.0f, 0.0f, 0.0f, 0.0f)
        : glm::make_quat(node.rotation.data());
    const glm::vec3 scale = node.scale.empty()
        ? glm::vec3(1.0f)
        : glm::make_vec3(node.scale.data());

    return utilityCore::buildTransformationMatrix(translation, rotation, scale);
}

const unsigned char* getAccessorElement(
    const tinygltf::Model& model,
    const tinygltf::Accessor& accessor,
    size_t index) {
    const tinygltf::BufferView& bufferView = model.bufferViews[accessor.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];
    const size_t stride = accessor.ByteStride(bufferView) > 0
        ? static_cast<size_t>(accessor.ByteStride(bufferView))
        : static_cast<size_t>(tinygltf::GetComponentSizeInBytes(accessor.componentType) *
              tinygltf::GetNumComponentsInType(accessor.type));
    return buffer.data.data() + bufferView.byteOffset + accessor.byteOffset + index * stride;
}

glm::vec2 readVec2(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t index) {
    const float* data = reinterpret_cast<const float*>(getAccessorElement(model, accessor, index));
    return glm::vec2(data[0], data[1]);
}

glm::vec3 readVec3(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t index) {
    const float* data = reinterpret_cast<const float*>(getAccessorElement(model, accessor, index));
    return glm::vec3(data[0], data[1], data[2]);
}

glm::vec4 readVec4(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t index) {
    const float* data = reinterpret_cast<const float*>(getAccessorElement(model, accessor, index));
    return glm::vec4(data[0], data[1], data[2], data[3]);
}

uint32_t readIndex(const tinygltf::Model& model, const tinygltf::Accessor& accessor, size_t index) {
    const unsigned char* data = getAccessorElement(model, accessor, index);
    switch (accessor.componentType) {
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
        return *reinterpret_cast<const uint8_t*>(data);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
        return *reinterpret_cast<const uint16_t*>(data);
    case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
        return *reinterpret_cast<const uint32_t*>(data);
    default:
        throw std::runtime_error("Unsupported glTF index component type.");
    }
}

void generateMissingTangents(RenderMesh& mesh) {
    if (!mesh.tangents.empty() ||
        mesh.positions.empty() ||
        mesh.normals.size() != mesh.positions.size() ||
        mesh.uvs.size() != mesh.positions.size()) {
        return;
    }

    std::vector<glm::vec3> tan1(mesh.positions.size(), glm::vec3(0.0f));
    std::vector<glm::vec3> tan2(mesh.positions.size(), glm::vec3(0.0f));

    const size_t triangleIndexCount = mesh.indices.size() - (mesh.indices.size() % 3);
    for (size_t i = 0; i < triangleIndexCount; i += 3) {
        const uint32_t i0 = mesh.indices[i + 0];
        const uint32_t i1 = mesh.indices[i + 1];
        const uint32_t i2 = mesh.indices[i + 2];
        if (i0 >= mesh.positions.size() || i1 >= mesh.positions.size() || i2 >= mesh.positions.size()) {
            continue;
        }

        const glm::vec3& v0 = mesh.positions[i0];
        const glm::vec3& v1 = mesh.positions[i1];
        const glm::vec3& v2 = mesh.positions[i2];
        const glm::vec2& uv0 = mesh.uvs[i0];
        const glm::vec2& uv1 = mesh.uvs[i1];
        const glm::vec2& uv2 = mesh.uvs[i2];

        const glm::vec3 dp1 = v1 - v0;
        const glm::vec3 dp2 = v2 - v0;
        const glm::vec2 du1 = uv1 - uv0;
        const glm::vec2 du2 = uv2 - uv0;
        const float denom = du1.x * du2.y - du2.x * du1.y;
        if (std::abs(denom) <= 1e-8f) {
            continue;
        }

        const float r = 1.0f / denom;
        const glm::vec3 sdir = (dp1 * du2.y - dp2 * du1.y) * r;
        const glm::vec3 tdir = (dp2 * du1.x - dp1 * du2.x) * r;

        tan1[i0] += sdir;
        tan1[i1] += sdir;
        tan1[i2] += sdir;
        tan2[i0] += tdir;
        tan2[i1] += tdir;
        tan2[i2] += tdir;
    }

    mesh.tangents.resize(mesh.positions.size(), glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    for (size_t i = 0; i < mesh.positions.size(); ++i) {
        const glm::vec3 normal = glm::normalize(mesh.normals[i]);
        glm::vec3 tangent = tan1[i] - normal * glm::dot(normal, tan1[i]);
        if (glm::dot(tangent, tangent) <= 1e-8f) {
            tangent = glm::abs(normal.y) < 0.999f
                ? glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), normal))
                : glm::normalize(glm::cross(glm::vec3(1.0f, 0.0f, 0.0f), normal));
        }
        else {
            tangent = glm::normalize(tangent);
        }

        const float handedness = glm::dot(glm::cross(normal, tangent), tan2[i]) < 0.0f ? -1.0f : 1.0f;
        mesh.tangents[i] = glm::vec4(tangent, handedness);
    }
}

RenderTextureRef toTextureRef(const tinygltf::TextureInfo& textureInfo) {
    RenderTextureRef ref;
    ref.textureId = textureInfo.index;
    ref.texCoord = textureInfo.texCoord;
    return ref;
}

RenderTextureRef toTextureRef(const tinygltf::NormalTextureInfo& textureInfo) {
    RenderTextureRef ref;
    ref.textureId = textureInfo.index;
    ref.texCoord = textureInfo.texCoord;
    return ref;
}

RenderMaterial convertMaterial(const tinygltf::Material& gltfMaterial) {
    RenderMaterial material;

    const auto& pbr = gltfMaterial.pbrMetallicRoughness;
    if (pbr.baseColorFactor.size() >= 3) {
        material.baseColor = glm::vec3(
            static_cast<float>(pbr.baseColorFactor[0]),
            static_cast<float>(pbr.baseColorFactor[1]),
            static_cast<float>(pbr.baseColorFactor[2]));
    }
    if (pbr.baseColorFactor.size() >= 4) {
        material.alpha = static_cast<float>(pbr.baseColorFactor[3]);
    }

    material.baseColorTexture = toTextureRef(pbr.baseColorTexture);
    material.metallic = static_cast<float>(pbr.metallicFactor);
    material.roughness = static_cast<float>(pbr.roughnessFactor);
    material.metallicRoughnessTexture = toTextureRef(pbr.metallicRoughnessTexture);

    if (gltfMaterial.emissiveFactor.size() >= 3) {
        material.emissiveColor = glm::vec3(
            static_cast<float>(gltfMaterial.emissiveFactor[0]),
            static_cast<float>(gltfMaterial.emissiveFactor[1]),
            static_cast<float>(gltfMaterial.emissiveFactor[2]));
        material.emissiveStrength = glm::length(material.emissiveColor) > 0.0f ? 1.0f : 0.0f;
    }
    material.emissiveTexture = toTextureRef(gltfMaterial.emissiveTexture);

    material.normalTexture = toTextureRef(gltfMaterial.normalTexture);
    material.normalScale = static_cast<float>(gltfMaterial.normalTexture.scale);
    material.doubleSided = gltfMaterial.doubleSided;
    material.useRoughSpecular = true;

    return material;
}

RenderMesh convertPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive) {
    if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
        throw std::runtime_error("Only triangle glTF primitives are supported.");
    }

    RenderMesh mesh;

    const auto positionIt = primitive.attributes.find("POSITION");
    if (positionIt == primitive.attributes.end()) {
        throw std::runtime_error("glTF primitive is missing POSITION data.");
    }

    const tinygltf::Accessor& positionAccessor = model.accessors[positionIt->second];
    mesh.positions.reserve(positionAccessor.count);
    for (size_t i = 0; i < positionAccessor.count; ++i) {
        mesh.positions.push_back(readVec3(model, positionAccessor, i));
    }

    const auto normalIt = primitive.attributes.find("NORMAL");
    if (normalIt != primitive.attributes.end()) {
        const tinygltf::Accessor& normalAccessor = model.accessors[normalIt->second];
        mesh.normals.reserve(normalAccessor.count);
        for (size_t i = 0; i < normalAccessor.count; ++i) {
            mesh.normals.push_back(readVec3(model, normalAccessor, i));
        }
    }

    const auto uvIt = primitive.attributes.find("TEXCOORD_0");
    if (uvIt != primitive.attributes.end()) {
        const tinygltf::Accessor& uvAccessor = model.accessors[uvIt->second];
        mesh.uvs.reserve(uvAccessor.count);
        for (size_t i = 0; i < uvAccessor.count; ++i) {
            mesh.uvs.push_back(readVec2(model, uvAccessor, i));
        }
    }

    const auto tangentIt = primitive.attributes.find("TANGENT");
    if (tangentIt != primitive.attributes.end()) {
        const tinygltf::Accessor& tangentAccessor = model.accessors[tangentIt->second];
        mesh.tangents.reserve(tangentAccessor.count);
        for (size_t i = 0; i < tangentAccessor.count; ++i) {
            mesh.tangents.push_back(readVec4(model, tangentAccessor, i));
        }
    }

    if (primitive.indices >= 0) {
        const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
        mesh.indices.reserve(indexAccessor.count);
        for (size_t i = 0; i < indexAccessor.count; ++i) {
            mesh.indices.push_back(readIndex(model, indexAccessor, i));
        }
    }
    else {
        mesh.indices.reserve(mesh.positions.size());
        for (uint32_t i = 0; i < mesh.positions.size(); ++i) {
            mesh.indices.push_back(i);
        }
    }

    generateMissingTangents(mesh);

    return mesh;
}

void appendTexture(RenderScene& scene, const tinygltf::Image& image) {
    RenderTexture texture;
    texture.width = image.width;
    texture.height = image.height;
    texture.components = image.component;
    texture.pixels = image.image;
    scene.textures.push_back(std::move(texture));
}

void addInstancesForNode(
    RenderScene& scene,
    const tinygltf::Model& model,
    const std::vector<std::vector<int>>& meshPrimitiveToRenderMesh,
    int nodeIndex,
    const glm::mat4& parentTransform,
    int defaultMaterialId) {
    const tinygltf::Node& node = model.nodes[nodeIndex];
    const glm::mat4 worldTransform = parentTransform * getNodeTransform(node);

    if (node.mesh >= 0) {
        const tinygltf::Mesh& gltfMesh = model.meshes[node.mesh];
        for (size_t primitiveIndex = 0; primitiveIndex < gltfMesh.primitives.size(); ++primitiveIndex) {
            const tinygltf::Primitive& primitive = gltfMesh.primitives[primitiveIndex];

            RenderInstance instance;
            instance.geometryType = RenderGeometryType::Mesh;
            instance.meshId = meshPrimitiveToRenderMesh[node.mesh][primitiveIndex];
            instance.materialId = primitive.material >= 0 ? primitive.material : defaultMaterialId;
            instance.transform = worldTransform;
            instance.inverseTransform = glm::inverse(worldTransform);
            instance.inverseTranspose = glm::inverseTranspose(worldTransform);

            const auto positionIt = primitive.attributes.find("POSITION");
            if (positionIt != primitive.attributes.end()) {
                const tinygltf::Accessor& positionAccessor = model.accessors[positionIt->second];
                if (positionAccessor.minValues.size() >= 3 && positionAccessor.maxValues.size() >= 3) {
                    instance.bboxMin = glm::vec3(
                        static_cast<float>(positionAccessor.minValues[0]),
                        static_cast<float>(positionAccessor.minValues[1]),
                        static_cast<float>(positionAccessor.minValues[2]));
                    instance.bboxMax = glm::vec3(
                        static_cast<float>(positionAccessor.maxValues[0]),
                        static_cast<float>(positionAccessor.maxValues[1]),
                        static_cast<float>(positionAccessor.maxValues[2]));
                }
            }

            scene.instances.push_back(instance);
        }
    }

    for (const int childIndex : node.children) {
        addInstancesForNode(
            scene,
            model,
            meshPrimitiveToRenderMesh,
            childIndex,
            worldTransform,
            defaultMaterialId);
    }
}

} // namespace

RenderScene GltfSceneLoader::load(const std::string& filename) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string error;
    std::string warning;

    const std::string extension = getFilePathExtension(filename);
    bool loaded = false;
    if (extension == "glb") {
        loaded = loader.LoadBinaryFromFile(&model, &error, &warning, filename);
    }
    else {
        loaded = loader.LoadASCIIFromFile(&model, &error, &warning, filename);
    }

    if (!warning.empty()) {
        std::cerr << "glTF parse warning: " << warning << std::endl;
    }
    if (!error.empty()) {
        std::cerr << "glTF parse error: " << error << std::endl;
    }
    if (!loaded) {
        throw std::runtime_error("Failed to load glTF scene: " + filename);
    }

    RenderScene scene;

    scene.materials.reserve(model.materials.size() + 1);
    for (const tinygltf::Material& material : model.materials) {
        scene.materials.push_back(convertMaterial(material));
    }
    const int defaultMaterialId = static_cast<int>(scene.materials.size());
    scene.materials.push_back(RenderMaterial{});

    scene.textures.reserve(model.textures.size());
    for (const tinygltf::Texture& texture : model.textures) {
        if (texture.source < 0 || texture.source >= static_cast<int>(model.images.size())) {
            scene.textures.push_back(RenderTexture{});
            continue;
        }
        appendTexture(scene, model.images[texture.source]);
    }

    std::vector<std::vector<int>> meshPrimitiveToRenderMesh(model.meshes.size());
    for (size_t meshIndex = 0; meshIndex < model.meshes.size(); ++meshIndex) {
        const tinygltf::Mesh& gltfMesh = model.meshes[meshIndex];
        meshPrimitiveToRenderMesh[meshIndex].reserve(gltfMesh.primitives.size());

        for (const tinygltf::Primitive& primitive : gltfMesh.primitives) {
            meshPrimitiveToRenderMesh[meshIndex].push_back(static_cast<int>(scene.meshes.size()));
            scene.meshes.push_back(convertPrimitive(model, primitive));
        }
    }

    if (model.defaultScene >= 0 && model.defaultScene < static_cast<int>(model.scenes.size())) {
        for (const int rootNode : model.scenes[model.defaultScene].nodes) {
            addInstancesForNode(
                scene,
                model,
                meshPrimitiveToRenderMesh,
                rootNode,
                glm::mat4(1.0f),
                defaultMaterialId);
        }
    }
    else {
        for (int nodeIndex = 0; nodeIndex < static_cast<int>(model.nodes.size()); ++nodeIndex) {
            addInstancesForNode(
                scene,
                model,
                meshPrimitiveToRenderMesh,
                nodeIndex,
                glm::mat4(1.0f),
                defaultMaterialId);
        }
    }

    return scene;
}
