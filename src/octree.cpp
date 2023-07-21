
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
            // If any of the three points of a triangle is within the bin,
            // then the whole triangle is part of the bin
            int n = _prim.i_offset + 3 * faceIdx;
            for (int idx = n; idx < n+3; ++idx)
            {
              const glm::vec3& pos = _vertices[_prim.v_offset + _indices[idx]];
              if (pos.x >= newBin.bbox_min.x && pos.x <= newBin.bbox_max.x &&
                pos.y >= newBin.bbox_min.y && pos.y <= newBin.bbox_max.y &&
                pos.z >= newBin.bbox_min.z && pos.z <= newBin.bbox_max.z) {
                // Add face to bin container
                facesInBin.push_back(faceIdx);
                break;
              }
            }
          }

          binFaces.push_back(facesInBin);

          // Check if cell needs further division
          std::cout << "Length " << glm::distance(newBin.bbox_max, newBin.bbox_min) << endl;
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
      std::cout << i << " size " << binFaces[i].size() << endl;
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
}