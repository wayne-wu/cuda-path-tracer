
#include <queue>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <unordered_set>

#include "octree.h"


Octree::Octree(const Primitive& prim, const Vertices& vertices, const Indices& indices) : 
  _prim(prim), _vertices(vertices), _indices(indices) {
  
  binDepths.push_back(0);

  binCorners.push_back(_prim.bbox_min);
  binCorners.push_back(_prim.bbox_max);

  binChildIndices.push_back(-1);

  divide();
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

    binMin = binCorners[2 * b];
    binMax = binCorners[2 * b + 1];

    cellSize = (binMax - binMin) / 2.0f;

    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {

          int newBin = binDepths.size();

          if (i == 0 && j == 0 && k == 0)
            binChildIndices[b] = newBin;

          int depth = binDepths[b] + 1;

          binDepths.push_back(depth);
          binChildIndices.push_back(-1);

          const glm::vec3 minCorner = binMin + glm::vec3(i, j, k) * cellSize;
          binCorners.push_back(minCorner);
          const glm::vec3 maxCorner = minCorner + cellSize;
          binCorners.push_back(maxCorner);

          vector<int> facesInBin;

          for (auto& faceIdx : binFaces[b]) {
            // If any of the three points of a triangle is within the bin,
            // then the whole triangle is part of the bin
            int n = _prim.i_offset + 3 * faceIdx;
            for (int idx = n; idx < n+3; ++idx)
            {
              const glm::vec3& pos = _vertices[_prim.v_offset + _indices[idx]];
              if (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
                pos.y >= minCorner.y && pos.y <= maxCorner.y &&
                pos.z >= minCorner.z && pos.z <= maxCorner.z) {
                // Add face to bin container
                facesInBin.push_back(faceIdx);
                break;
              }
            }
          }

          binFaces.push_back(facesInBin);

          // Check if cell needs further division
          if (length2(maxCorner - minCorner) > minCellsize && facesInBin.size() > minBinCount && depth < maxDepth)
            divideQueue.push(newBin);
        }
      }
    }

    binFaces[b].clear();  // Clear all faces within bin after dividing
  }

  binStartIndices = vector<int>(binChildIndices.size(), -1);
  binEndIndices = vector<int>(binChildIndices.size(), -1);

  for (int i = 0; i < binFaces.size(); ++i) {
    if (!binFaces[i].empty()) {
      binStartIndices[i] = faceBins.size();
      faceBins.insert(faceBins.end(), binFaces[i].begin(), binFaces[i].end());
      binEndIndices[i] = faceBins.size();
    }
  }

  unordered_set<int> binnedFaces(faceBins.begin(), faceBins.end());
  // Assert that all faces are part of a bin
  assert(binnedFaces.size() == numFaces);
}