#include "TextSceneLoader.h"

#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>

#include "GltfSceneLoader.h"
#include "core/utilities.h"
#include "geometry/octree.h"
#include "stb_image.h"

namespace {

std::string toUpper(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return value;
}

std::string resolvePathFromSceneFile(const std::string& sceneFilename, const std::string& assetPath) {
    namespace fs = std::filesystem;

    fs::path asset(assetPath);
    if (asset.is_absolute()) {
        return fs::weakly_canonical(asset).string();
    }

    fs::path scenePath(sceneFilename);
    fs::path sceneDir = scenePath.has_parent_path() ? scenePath.parent_path() : fs::current_path();
    return fs::weakly_canonical(sceneDir / asset).string();
}

glm::vec3 readVec3(const std::vector<std::string>& tokens) {
    return glm::vec3(
        std::atof(tokens[1].c_str()),
        std::atof(tokens[2].c_str()),
        std::atof(tokens[3].c_str()));
}

void offsetTextureRef(RenderTextureRef& ref, int textureOffset) {
    if (ref.valid()) {
        ref.textureId += textureOffset;
    }
}

void appendGltfScene(RenderScene& scene, RenderScene gltfScene) {
    const int meshOffset = static_cast<int>(scene.meshes.size());
    const int materialOffset = static_cast<int>(scene.materials.size());
    const int textureOffset = static_cast<int>(scene.textures.size());

    scene.textures.insert(
        scene.textures.end(),
        std::make_move_iterator(gltfScene.textures.begin()),
        std::make_move_iterator(gltfScene.textures.end()));

    for (RenderMaterial& material : gltfScene.materials) {
        offsetTextureRef(material.baseColorTexture, textureOffset);
        offsetTextureRef(material.metallicRoughnessTexture, textureOffset);
        offsetTextureRef(material.emissiveTexture, textureOffset);
        offsetTextureRef(material.normalTexture, textureOffset);
        scene.materials.push_back(std::move(material));
    }

    scene.meshes.insert(
        scene.meshes.end(),
        std::make_move_iterator(gltfScene.meshes.begin()),
        std::make_move_iterator(gltfScene.meshes.end()));

    for (RenderInstance& instance : gltfScene.instances) {
        if (instance.meshId >= 0) {
            instance.meshId += meshOffset;
        }
        if (instance.materialId >= 0) {
            instance.materialId += materialOffset;
        }
        scene.instances.push_back(instance);
    }
}

void loadMaterial(std::ifstream& file, RenderScene& scene, const std::string& materialId) {
    const int id = std::atoi(materialId.c_str());
    if (id != static_cast<int>(scene.materials.size())) {
        throw std::runtime_error("MATERIAL ID does not match expected number of materials.");
    }

    std::cout << "Loading Material " << id << "..." << std::endl;

    RenderMaterial material;
    glm::vec3 color(1.0f);

    for (int i = 0; i < 7; ++i) {
        std::string line;
        utilityCore::safeGetline(file, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens.empty()) {
            continue;
        }

        if (tokens[0] == "RGB") {
            color = readVec3(tokens);
            material.baseColor = color;
        }
        else if (tokens[0] == "REFL") {
            const float reflectivity = std::atof(tokens[1].c_str());
            material.metallic = reflectivity;
            material.roughness = 0.0f;
        }
        else if (tokens[0] == "REFR") {
            material.transmission = std::atof(tokens[1].c_str());
        }
        else if (tokens[0] == "REFRIOR") {
            material.ior = std::atof(tokens[1].c_str());
        }
        else if (tokens[0] == "EMITTANCE") {
            material.emissiveStrength = std::atof(tokens[1].c_str());
            material.emissiveColor = color;
        }
    }

    scene.materials.push_back(material);
}

void loadObject(std::ifstream& file, RenderScene& scene, const std::string& objectId) {
    const int id = std::atoi(objectId.c_str());
    if (id != static_cast<int>(scene.instances.size())) {
        throw std::runtime_error("OBJECT ID does not match expected number of objects.");
    }

    std::cout << "Loading Object " << id << "..." << std::endl;

    RenderInstance instance;
    std::string line;
    utilityCore::safeGetline(file, line);
    if (line == "sphere") {
        instance.geometryType = RenderGeometryType::Sphere;
    }
    else if (line == "cube") {
        instance.geometryType = RenderGeometryType::Cube;
    }
    else {
        throw std::runtime_error("Unsupported text scene object type: " + line);
    }

    utilityCore::safeGetline(file, line);
    std::vector<std::string> tokens = utilityCore::tokenizeString(line);
    if (tokens.size() >= 2 && toUpper(tokens[0]) == "MATERIAL") {
        instance.materialId = std::atoi(tokens[1].c_str());
    }

    glm::vec3 translation(0.0f);
    glm::vec3 rotation(0.0f);
    glm::vec3 scale(1.0f);

    utilityCore::safeGetline(file, line);
    while (!line.empty() && file.good()) {
        tokens = utilityCore::tokenizeString(line);
        if (!tokens.empty()) {
            if (tokens[0] == "TRANS") {
                translation = readVec3(tokens);
            }
            else if (tokens[0] == "ROTAT") {
                rotation = readVec3(tokens);
            }
            else if (tokens[0] == "SCALE") {
                scale = readVec3(tokens);
            }
        }
        utilityCore::safeGetline(file, line);
    }

    instance.transform = utilityCore::buildTransformationMatrix(translation, rotation, scale);
    instance.inverseTransform = glm::inverse(instance.transform);
    instance.inverseTranspose = glm::inverseTranspose(instance.transform);
    instance.bboxMin = glm::vec3(-0.5f);
    instance.bboxMax = glm::vec3(0.5f);

    scene.instances.push_back(instance);
}

void loadCamera(std::ifstream& file, RenderScene& scene) {
    std::cout << "Loading Camera ..." << std::endl;

    float fovy = 45.0f;

    for (int i = 0; i < 5; ++i) {
        std::string line;
        utilityCore::safeGetline(file, line);
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens.empty()) {
            continue;
        }

        if (tokens[0] == "RES") {
            scene.camera.resolution.x = std::atoi(tokens[1].c_str());
            scene.camera.resolution.y = std::atoi(tokens[2].c_str());
        }
        else if (tokens[0] == "FOVY") {
            fovy = std::atof(tokens[1].c_str());
            scene.camera.verticalFovDegrees = fovy;
        }
        else if (tokens[0] == "ITERATIONS") {
            scene.settings.iterations = std::atoi(tokens[1].c_str());
        }
        else if (tokens[0] == "DEPTH") {
            scene.settings.traceDepth = std::atoi(tokens[1].c_str());
        }
        else if (tokens[0] == "FILE") {
            scene.settings.imageName = tokens[1];
        }
    }

    std::string line;
    utilityCore::safeGetline(file, line);
    while (!line.empty() && file.good()) {
        std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (!tokens.empty()) {
            if (tokens[0] == "EYE") {
                scene.camera.position = readVec3(tokens);
            }
            else if (tokens[0] == "LOOKAT") {
                scene.camera.lookAt = readVec3(tokens);
            }
            else if (tokens[0] == "UP") {
                scene.camera.up = readVec3(tokens);
            }
        }
        utilityCore::safeGetline(file, line);
    }

    scene.camera.verticalFovDegrees = fovy;
    std::cout << "Loaded camera!" << std::endl;
}

void loadEnvironment(std::ifstream& file, RenderScene& scene, const std::string& sceneFilename) {
    std::cout << "Loading Environment ..." << std::endl;

    std::string imagePath;
    std::string line;
    utilityCore::safeGetline(file, line);
    while (!line.empty() && file.good()) {
        const std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens.size() >= 2) {
            if (tokens[0] == "FILE") {
                imagePath = resolvePathFromSceneFile(sceneFilename, tokens[1]);
            }
            else if (tokens[0] == "INTENSITY") {
                scene.environment.intensity = std::atof(tokens[1].c_str());
            }
            else if (tokens[0] == "ROTATION") {
                scene.environment.rotation = std::atof(tokens[1].c_str()) * PI / 180.0f;
            }
        }
        utilityCore::safeGetline(file, line);
    }

    if (imagePath.empty()) {
        throw std::runtime_error("ENVIRONMENT requires FILE.");
    }

    int width = 0;
    int height = 0;
    int components = 0;
    float* image = stbi_loadf(imagePath.c_str(), &width, &height, &components, 4);
    if (image == nullptr) {
        throw std::runtime_error("Failed to load environment map: " + imagePath);
    }

    scene.environment.width = width;
    scene.environment.height = height;
    scene.environment.enabled = true;
    scene.environment.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height));
    for (int i = 0; i < width * height; ++i) {
        scene.environment.pixels[i] = glm::vec4(
            image[4 * i + 0],
            image[4 * i + 1],
            image[4 * i + 2],
            image[4 * i + 3]);
    }

    stbi_image_free(image);
    std::cout << "Loaded environment map " << imagePath << " (" << width << "x" << height << ")" << std::endl;
}

glm::vec3 computeMeshMin(const RenderMesh& mesh) {
    glm::vec3 value = mesh.positions.front();
    for (const glm::vec3& position : mesh.positions) {
        value = glm::min(value, position);
    }
    return value;
}

glm::vec3 computeMeshMax(const RenderMesh& mesh) {
    glm::vec3 value = mesh.positions.front();
    for (const glm::vec3& position : mesh.positions) {
        value = glm::max(value, position);
    }
    return value;
}

void buildOctrees(RenderScene& scene) {
    std::cout << "\nConstructing Octrees ..." << std::endl;
    for (RenderMesh& mesh : scene.meshes) {
        if (mesh.positions.empty() || mesh.indices.empty()) {
            continue;
        }

        Octree octree(mesh, computeMeshMin(mesh), computeMeshMax(mesh));
        mesh.bins = std::move(octree.bins);
        mesh.binFaces = std::move(octree.faceBins);
    }
    std::cout << "Octrees completed." << std::endl << std::endl;
}

} // namespace

RenderScene TextSceneLoader::load(const std::string& filename) {
    std::cout << "Reading scene from " << filename << " ..." << std::endl << std::endl;

    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Error reading scene file: " + filename);
    }

    RenderScene scene;

    while (file.good()) {
        std::string line;
        utilityCore::safeGetline(file, line);
        if (line.empty()) {
            continue;
        }

        const std::vector<std::string> tokens = utilityCore::tokenizeString(line);
        if (tokens.empty()) {
            continue;
        }

        if (tokens[0] == "MATERIAL") {
            loadMaterial(file, scene, tokens[1]);
            std::cout << std::endl;
        }
        else if (tokens[0] == "OBJECT") {
            loadObject(file, scene, tokens[1]);
            std::cout << std::endl;
        }
        else if (tokens[0] == "CAMERA") {
            loadCamera(file, scene);
            std::cout << std::endl;
        }
        else if (tokens[0] == "GLTF") {
            GltfSceneLoader loader;
            RenderScene gltfScene = loader.load(resolvePathFromSceneFile(filename, tokens[1]));
            appendGltfScene(scene, std::move(gltfScene));
            std::cout << std::endl;
        }
        else if (tokens[0] == "ENVIRONMENT") {
            loadEnvironment(file, scene, filename);
            std::cout << std::endl;
        }
    }

#if OCTREE
    buildOctrees(scene);
#endif

    return scene;
}
