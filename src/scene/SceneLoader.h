#pragma once

#include <string>
#include <filesystem>

#include "RenderScene.h"

static std::string GetFilePathExtension(const std::string& FileName) {
  if (FileName.find_last_of(".") != std::string::npos)
    return FileName.substr(FileName.find_last_of(".") + 1);
  return "";
}

static std::string ResolvePathFromSceneFile(const std::string& sceneFilename, const std::string& assetPath) {
  namespace fs = std::filesystem;

  fs::path asset(assetPath);
  if (asset.is_absolute()) {
    return fs::weakly_canonical(asset).string();
  }

  fs::path scenePath(sceneFilename);
  fs::path sceneDir = scenePath.has_parent_path() ? scenePath.parent_path() : fs::current_path();
  return fs::weakly_canonical(sceneDir / asset).string();
}

class SceneLoader {
public:
    virtual ~SceneLoader() = default;
    virtual RenderScene load(const std::string& filename) = 0;
};
