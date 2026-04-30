
#include <queue>
#include <algorithm>
#include <cassert>
#include <numeric>
#include <iostream>
#include <unordered_set>

#include "octree.h"


Octree::Octree(const RenderMesh& mesh, const glm::vec3& bboxMin, const glm::vec3& bboxMax) :
  _mesh(mesh), _bboxMin(bboxMin), _bboxMax(bboxMax) {
  
  RenderBin root;
  root.depth = 0;
  root.bboxMax = _bboxMax;
  root.bboxMin = _bboxMin;
  root.childIndex = -1;

  bins.push_back(root);

  divide();
}

void Octree::project(const std::vector<glm::vec3>& points, const glm::vec3& axis, float& minVal, float& maxVal) {

  minVal = INFINITY;
  maxVal = -INFINITY;
  for (auto& p : points)
  {
    float val = glm::dot(axis, p);
    minVal = std::min(minVal , val);
    maxVal = std::max(maxVal, val);
  }
}

/*
Adapted from: https://stackoverflow.com/questions/17458562/efficient-aabb-triangle-intersection-in-c-sharp
*/
bool Octree::intersectAABBTriangle(const glm::vec3& bboxMin, const glm::vec3& bboxMax, std::vector<glm::vec3>& v) {

  std::vector<glm::vec3> boxVertices = {
    bboxMin,
    glm::vec3(bboxMin.x, bboxMin.y, bboxMax.z),
    glm::vec3(bboxMin.x, bboxMax.y, bboxMax.z),
    glm::vec3(bboxMin.x, bboxMax.y, bboxMin.z),
    glm::vec3(bboxMax.x, bboxMax.y, bboxMin.z),
    glm::vec3(bboxMax.x, bboxMin.y, bboxMax.z),
    glm::vec3(bboxMax.x, bboxMin.y, bboxMin.z),
    bboxMax,
  };

  std::vector<glm::vec3> boxNormals = {
    glm::vec3(1,0,0), glm::vec3(0,1,0), glm::vec3(0,0,1)
  };

  float min, max;
  for (int i = 0; i < 3; ++i)
  {
    project(v, boxNormals[i], min, max);
    if (max < bboxMin[i] || min > bboxMax[i])
      return false;
  }

  std::vector<glm::vec3> edges = {
    v[0] - v[1], v[1] - v[2], v[2] - v[0]
  };

  glm::vec3 normal = -glm::normalize(glm::cross(v[0]-v[1], v[2]-v[1]));

  float triangleOffset = glm::dot(normal, v[1]);
  project(boxVertices, normal, min, max);
  if (max < triangleOffset || min > triangleOffset)
    return false;

  float min2, max2;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      glm::vec3 a = glm::cross(edges[i], boxNormals[j]);
      project(boxVertices, a, min, max);
      project(v, a, min2, max2);
      if (max < min2 || min > max2)
        return false;
    }
  }

  return true;
}

void Octree::divide() {

  size_t numFaces = _mesh.indices.size() / 3;

  std::vector<std::vector<int>> binFaces;

  std::vector<int> faces(numFaces);
  std::iota(faces.begin(), faces.end(), 0);
  binFaces.push_back(faces);

  std::queue<int> divideQueue;
  divideQueue.push(0);

  glm::vec3 binMin, binMax, cellSize;

  while (!divideQueue.empty()) {

    int b = divideQueue.front(); divideQueue.pop();

    RenderBin bin = bins[b];

    cellSize = (bin.bboxMax - bin.bboxMin) / 2.0f;

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {

          int newBinIdx = bins.size();

          if (i == 0 && j == 0 && k == 0)
            bins[b].childIndex = newBinIdx;

          RenderBin newBin;
          newBin.depth = bin.depth + 1;
          newBin.childIndex = -1;
          newBin.bboxMin = bin.bboxMin + glm::vec3(i, j, k) * cellSize;
          newBin.bboxMax = newBin.bboxMin + cellSize;
          bins.push_back(newBin);

          std::vector<int> facesInBin;

          for (auto& faceIdx : binFaces[b]) {
            // Checks if the triangle intersects with the bin
            // Even if the three points are not inside it
            int n = 3 * faceIdx;

            std::vector<glm::vec3> v;
            for (int idx = n; idx < n+3; ++idx)
              v.push_back(_mesh.positions[_mesh.indices[idx]]);

            if (intersectAABBTriangle(newBin.bboxMin, newBin.bboxMax, v))
              facesInBin.push_back(faceIdx);
          }

          binFaces.push_back(facesInBin);

          // Check if cell needs further division
          // std::cout << "Length " << glm::distance(newBin.bbox_max, newBin.bbox_min) << endl;
          if (glm::distance(newBin.bboxMax, newBin.bboxMin) > minCellsize && 
            facesInBin.size() > minBinCount && 
            newBin.depth < maxDepth)
            divideQueue.push(newBinIdx);
        }
      }
    }

    binFaces[b].clear();  // Clear all faces within bin after dividing
  }

  for (int i = 0; i < binFaces.size(); ++i) {
    if (!binFaces[i].empty()) {
      // std::cout << i << " size " << binFaces[i].size() << endl;
      bins[i].startIndex = faceBins.size();
      faceBins.insert(faceBins.end(), binFaces[i].begin(), binFaces[i].end());
      bins[i].endIndex = faceBins.size();
    }
    else {
      bins[i].startIndex = -1;
      bins[i].endIndex = -1;
    }
  }

  std::unordered_set<int> binnedFaces(faceBins.begin(), faceBins.end());
  // Assert that all faces are part of a bin
  assert(binnedFaces.size() == numFaces);

  std::cout << "Final Depth: " << bins.back().depth << std::endl;
}
