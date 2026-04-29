
#include <queue>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_set>

#include "octree.h"


Octree::Octree(const Primitive& prim, const Vertices& vertices, const Indices& indices) : 
  _prim(prim), _vertices(vertices), _indices(indices) {
  
  Bin root;
  root.depth = 0;
  root.bbox_max = _prim.bbox_max;
  root.bbox_min = _prim.bbox_min;
  root.childIndex = -1;

  bins.push_back(root);

  divide();
}

void Octree::project(vector<Vec3> points, Vec3 axis, float& minVal, float& maxVal) {

  minVal = INFINITY;
  maxVal = -INFINITY;
  for (auto& p : points)
  {
    float val = glm::dot(axis, p);
    minVal = min(minVal , val);
    maxVal = max(maxVal, val);
  }
}

/*
Adapted from: https://stackoverflow.com/questions/17458562/efficient-aabb-triangle-intersection-in-c-sharp
*/
bool Octree::intersectAABBTriangle(Vec3& bboxMin, Vec3& bboxMax, vector<Vec3>& v) {

  vector<Vec3> boxVertices = {
    bboxMin,
    Vec3(bboxMin.x, bboxMin.y, bboxMax.z),
    Vec3(bboxMin.x, bboxMax.y, bboxMax.z),
    Vec3(bboxMin.x, bboxMax.y, bboxMin.z),
    Vec3(bboxMax.x, bboxMax.y, bboxMin.z),
    Vec3(bboxMax.x, bboxMin.y, bboxMax.z),
    Vec3(bboxMax.x, bboxMin.y, bboxMin.z),
    bboxMax,
  };

  vector<Vec3> boxNormals = {
    Vec3(1,0,0), Vec3(0,1,0), Vec3(0,0,1)
  };

  float min, max;
  for (int i = 0; i < 3; ++i)
  {
    project(v, boxNormals[i], min, max);
    if (max < bboxMin[i] || min > bboxMax[i])
      return false;
  }

  vector<Vec3> edges = {
    v[0] - v[1], v[1] - v[2], v[2] - v[0]
  };

  Vec3 normal = -glm::normalize(glm::cross(v[0]-v[1], v[2]-v[1]));

  float triangleOffset = glm::dot(normal, v[1]);
  project(boxVertices, normal, min, max);
  if (max < triangleOffset || min > triangleOffset)
    return false;

  float min2, max2;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      Vec3 a = glm::cross(edges[i], boxNormals[j]);
      project(boxVertices, a, min, max);
      project(v, a, min2, max2);
      if (max < min2 || min > max2)
        return false;
    }
  }

  return true;
}

void Octree::divide() {

  size_t numFaces = _prim.count / 3;

  vector<vector<int>> binFaces;

  vector<int> faces(numFaces);
  std::iota(faces.begin(), faces.end(), 0);
  binFaces.push_back(faces);

  queue<int> divideQueue;
  divideQueue.push(0);

  glm::vec3 binMin, binMax, cellSize;

  while (!divideQueue.empty()) {

    int b = divideQueue.front(); divideQueue.pop();

    Bin bin = bins[b];

    cellSize = (bin.bbox_max - bin.bbox_min) / 2.0f;

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {

          int newBinIdx = bins.size();

          if (i == 0 && j == 0 && k == 0)
            bins[b].childIndex = newBinIdx;

          Bin newBin;
          newBin.depth = bin.depth + 1;
          newBin.childIndex = -1;
          newBin.bbox_min = bin.bbox_min + Vec3(i, j, k) * cellSize;
          newBin.bbox_max = newBin.bbox_min + cellSize;
          bins.push_back(newBin);

          vector<int> facesInBin;

          for (auto& faceIdx : binFaces[b]) {
            // Checks if the triangle intersects with the bin
            // Even if the three points are not inside it
            int n = _prim.i_offset + 3 * faceIdx;

            vector<Vec3> v;
            for (int idx = n; idx < n+3; ++idx)
              v.push_back(_vertices[_prim.v_offset + _indices[idx]]);

            if (intersectAABBTriangle(newBin.bbox_min, newBin.bbox_max, v))
              facesInBin.push_back(faceIdx);
          }

          binFaces.push_back(facesInBin);

          // Check if cell needs further division
          // std::cout << "Length " << glm::distance(newBin.bbox_max, newBin.bbox_min) << endl;
          if (glm::distance(newBin.bbox_max, newBin.bbox_min) > minCellsize && 
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

  unordered_set<int> binnedFaces(faceBins.begin(), faceBins.end());
  // Assert that all faces are part of a bin
  assert(binnedFaces.size() == numFaces);

  std::cout << "Final Depth: " << bins.back().depth << std::endl;
}