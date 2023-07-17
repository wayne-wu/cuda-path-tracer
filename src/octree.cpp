
#include <queue>
#include <algorithm>

#include "octree.h"
#include "main.h"


Octree::Octree(const Primitive& prim, const Vertices& vertices, const Indices& indices) : 
  _prim(prim), _vertices(vertices), _indices(indices) {
  
  binDepths.push_back(0);
  binParents.push_back(0);

  binCorners.push_back(_prim.bbox_min);
  binCorners.push_back(_prim.bbox_max);

  binChildIndices.push_back(-1);

  triBins = vector<int>(_prim.count/3, 0);

  divide(0);
}

void Octree::divide(int bin) {

  queue<int> divideQueue;
  divideQueue.push(bin);

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
          binParents.push_back(b);
          binChildIndices.push_back(-1);

          binCorners.emplace_back(binMin + glm::vec3(i, j, k) * cellSize);
          const glm::vec3& minCorner = binCorners.back();
          binCorners.emplace_back(minCorner + cellSize);
          const glm::vec3& maxCorner = binCorners.back();

          int count = 0;

          for (int primnum = 0; primnum < triBins.size(); ++primnum) {
            if (triBins[primnum] != b)
              continue;

            bool inBin = false;

            int idx = _prim.i_offset + 3 * primnum;

            // If any of the three points of a triangle is within the bin,
            // then this triangle is part of the bin
            for (int inc = 0; inc < 3; ++inc)
            {
              int vIdx = _indices[idx + inc];
              const glm::vec3& pos = _vertices[_prim.v_offset + vIdx];
              if (pos.x >= minCorner.x && pos.x <= maxCorner.x &&
                pos.y >= minCorner.y && pos.y <= maxCorner.y &&
                pos.z >= minCorner.z && pos.z <= maxCorner.z) {
                inBin = true;
                break;
              }
            }

            if (inBin) {
              triBins[(idx - _prim.i_offset)/3] = newBin;
              // pointBins[ptnum] = pointBins[ptnum + 1] = pointBins[ptnum + 2] = newBin;
              ++count;
            }
          }

          if (length2(maxCorner - minCorner) > minCellsize && count > minBinCount && depth < maxDepth)
          {
            divideQueue.push(newBin);
          }
        }
      }
    }
  }

  // Sort triBins
  sort(triBins.begin(), triBins.end());

  binStartIndices = vector<int>(binChildIndices.size(), -1);
  binEndIndices = vector<int>(binChildIndices.size(), -1);

  binStartIndices[triBins[0]] = 0;

  for (int i = 1; i < triBins.size(); ++i) {
    if (triBins[i] != triBins[i - 1]) {
      binEndIndices[triBins[i - 1]] = i - 1;
      binStartIndices[triBins[i]] = i;
    }
  }

}