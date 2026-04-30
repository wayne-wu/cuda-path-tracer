#pragma once

#include "SceneLoader.h"

class GltfSceneLoader : public SceneLoader {
public:
    RenderScene load(const std::string& filename) override;
};

