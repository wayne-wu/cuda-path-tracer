#pragma once

#include <vector>
#include "glm/glm.hpp"

#include "core/utilities.h"
#include "scene/RenderScene.h"


class Octree {
  
private:
  void divide();
  void project(const std::vector<glm::vec3>& points, const glm::vec3& axis, float& min, float& max);
  bool intersectAABBTriangle(const glm::vec3& center, const glm::vec3& extents, std::vector<glm::vec3>& v);

  const RenderMesh& _mesh;
  glm::vec3 _bboxMin;
  glm::vec3 _bboxMax;

  int minBinCount = 100;
  int maxDepth = 6;
  float minCellsize = 0.01f;

public:
  Octree(const RenderMesh& mesh, const glm::vec3& bboxMin, const glm::vec3& bboxMax);
  ~Octree() {};


  std::vector<RenderBin> bins;


  // Stores the bin number for all vertices (all triangles)
  std::vector<int> faceBins;
};
