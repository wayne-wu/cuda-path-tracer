#pragma once

#include <vector>
#include "glm/glm.hpp"

#include "utilities.h"
#include "sceneStructs.h"


using namespace std;

class Octree {
  
private:
  void divide();

  // TODO: Figure out the best way to pass in scene data
  const Primitive& _prim;
  const Vertices& _vertices;

  const Indices& _indices;

  int minBinCount = 100;
  int maxDepth = 10;
  float minCellsize = 0.01f;

public:
  Octree::Octree(const Primitive& prim, const Vertices& vertices, const Indices& indices);
  Octree::~Octree() {};


  vector<Bin> bins;


  // Stores the bin number for all vertices (all triangles)
  vector<int> faceBins;
};
