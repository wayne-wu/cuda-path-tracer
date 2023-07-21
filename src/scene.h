#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "octree.h"

#include "tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    int loadMaterial(string materialid);
    int loadGeom(string objectid);
    int loadCamera();
    int loadGLTFNode(const std::vector<tinygltf::Node>& nodes, 
      const tinygltf::Node& node, const glm::mat4& xform, bool* isLoaded);
    int loadGLTF(const std::string& filename, float scale);
public:
    Scene(string filename);
    ~Scene() {};

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<Texture> textures;

    std::vector<Mesh>      meshes;
    std::vector<Primitive> primitives;

    // Primitive Data
    Indices  mesh_indices;
    Vertices mesh_vertices;
    std::vector<Vec3> mesh_normals;
    std::vector<Vec2> mesh_uvs;
    std::vector<Vec4> mesh_tangents;

    std::vector<Bin> bins;
    std::vector<int> faceBins;

    RenderState state;
};
