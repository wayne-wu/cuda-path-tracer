#pragma once

#include "SceneLoader.h"

class TextSceneLoader : public SceneLoader {
public:
    RenderScene load(const std::string& filename) override;
};

