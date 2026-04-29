#pragma once

#include <vector>
#include "glm/glm.hpp"

#include "utilities.h"
#include "sceneStructs.h"


using namespace std;

class Octree {
  
private:
  void divide();
  void project(vector<Vec3> points, Vec3 axis, float& min, float& max);
  bool intersectAABBTriangle(Vec3& center, Vec3& extents, vector<Vec3>& v);

  // TODO: Figure out the best way to pass in scene data
  const Primitive& _prim;
  const Vertices& _vertices;

  const Indices& _indices;

  int minBinCount = 100;
  int maxDepth = 6;
  float minCellsize = 0.01f;

public:
  Octree::Octree(const Primitive& prim, const Vertices& vertices, const Indices& indices);
  Octree::~Octree() {};


  vector<Bin> bins;


  // Stores the bin number for all vertices (all triangles)
  vector<int> faceBins;
};
