#pragma once

#include <vector>
#include "glm/glm.hpp"

#include "sceneStructs.h"


using namespace std;

class Octree {
  
private:
  void divide(int bin);

  // TODO: Figure out the best way to pass in scene data
  const Primitive& _prim;
  const Vertices& _vertices;
  const Indices& _indices;

  int minBinCount = 20;
  int maxDepth = 10;
  float minCellsize = 0.1f;

public:
  Octree::Octree(const Primitive& prim, const Vertices& vertices, const Indices& indices);
  Octree::~Octree() {};

  vector<int> binDepths;
  vector<int> binParents;

  // Stores the min and max corners for a bin
  vector<glm::vec3> binCorners;
  // Stores the starting index for the child of a bin.
  // -1 if the bin is a leaf node
  vector<int> binChildIndices;
  
  vector<int> binStartIndices;
  vector<int> binEndIndices;

  // Stores the bin number for all vertices (all triangles)
  vector<int> triBins;
};
