#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include "core/glslUtility.hpp"
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <memory>

#include "core/image.h"
#include "core/utilities.h"

#include "scene/RenderScene.h"
#include "scene/GltfSceneLoader.h"
#include "renderer/PathTracer.h"

using namespace std;

//-------------------------------
//----------PATH TRACER----------
//-------------------------------

extern std::unique_ptr<RenderScene> scene;
extern std::unique_ptr<PathTracer> pathTracer;
extern int iteration;

extern int width;
extern int height;

extern int ui_iterations;
extern int startupIterations;
extern bool ui_showGbuffer;
extern bool ui_denoise;
extern int ui_filterSize;
extern int ui_filterPasses;
extern float ui_colorWeight;
extern float ui_normalWeight;
extern float ui_positionWeight;
extern bool ui_saveAndExit;
extern double ui_autoExitSeconds;
extern string ui_sceneFile;

extern cudaGraphicsResource_t pbo_resource;

void loadScene(string scene);
void runApp();
void runCuda();
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods);
void mousePositionCallback(GLFWwindow* window, double xpos, double ypos);
void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
