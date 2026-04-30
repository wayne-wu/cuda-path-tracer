#include "main.h"
#include "preview.h"
#include "scene/TextSceneLoader.h"
#include <cstring>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

int ui_iterations = 100;
int startupIterations = 100;
int lastLoopIterations = 0;
string lastSceneFile = "";

bool ui_showGbuffer = false;
int ui_GbufferMode = GBUFFER_NORMAL;  //switch between different gbuffers
bool ui_denoise = true;
bool ui_saveAndExit = false;
double ui_autoExitSeconds = -1.0;
string ui_sceneFile = "";

cudaGraphicsResource_t pbo_resource;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

std::unique_ptr<RenderScene> scene;
std::unique_ptr<PathTracer> pathTracer;
int iteration;

int width = 800;
int height = 800;

struct CameraBasis {
    glm::vec3 view;
    glm::vec3 right;
    glm::vec3 up;
};

static CameraBasis computeCameraBasis(const RenderCamera& camera) {
    CameraBasis basis = {};

    glm::vec3 view = camera.lookAt - camera.position;
    if (glm::length(view) == 0.0f) {
        view = glm::vec3(0.0f, 0.0f, -1.0f);
    }
    basis.view = glm::normalize(view);

    glm::vec3 up = camera.up;
    if (glm::length(up) == 0.0f) {
        up = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    up = glm::normalize(up);

    glm::vec3 right = glm::cross(basis.view, up);
    if (glm::length(right) == 0.0f) {
        right = glm::cross(basis.view, glm::vec3(0.0f, 0.0f, 1.0f));
    }
    if (glm::length(right) == 0.0f) {
        right = glm::vec3(1.0f, 0.0f, 0.0f);
    }

    basis.right = glm::normalize(right);
    basis.up = glm::normalize(glm::cross(basis.right, basis.view));
    return basis;
}

static int triangleCount(const RenderScene& renderScene) {
    int count = 0;
    for (const RenderMesh& mesh : renderScene.meshes) {
        count += static_cast<int>(mesh.indices.size() / 3);
    }
    return count;
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
    startTimeString = currentTimeString();

    for (int i = 1; i < argc; ++i) {
      if (std::strcmp(argv[i], "--auto-exit-seconds") == 0) {
        if (i + 1 >= argc) {
          std::cerr << "Missing value for --auto-exit-seconds" << std::endl;
          return EXIT_FAILURE;
        }
        ui_autoExitSeconds = std::atof(argv[++i]);
      }
      else if (ui_sceneFile.empty()) {
        loadScene(argv[i]);
        lastSceneFile = ui_sceneFile = argv[i];
      }
      else {
        std::cerr << "Unrecognized argument: " << argv[i] << std::endl;
        return EXIT_FAILURE;
      }
    }

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

      return 0;
}


void loadScene(string sceneFile) {
    // Load scene file
    TextSceneLoader loader;
    scene = std::make_unique<RenderScene>(loader.load(sceneFile));

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    RenderCamera &cam = scene->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;
    // ui_iterations = static_cast<int>(scene->settings.iterations);
    startupIterations = ui_iterations;

    CameraBasis basis = computeCameraBasis(cam);
    glm::vec3 view = basis.view;

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    lastSceneFile = sceneFile;
}

void saveImage() {
    const bool useDenoised = pathTracer && pathTracer->hasDenoisedImage();
    const int samples = iteration > 0 ? iteration : 1;

    // output image file
    image img(width, height);
    std::vector<glm::vec3> pixels;
    pathTracer->copyImageToHost(pixels, useDenoised);

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {
            int index = x + (y * width);
            glm::vec3 pix = index < pixels.size() ? pixels[index] : glm::vec3(0.0f);
            img.setPixel(width - 1 - x, y, pix);
        }
    }

    std::string filename = scene ? scene->settings.imageName : std::string("render");
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp";
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda() {

    if (lastSceneFile != ui_sceneFile) {
      int oldWidth = width;
      int oldHeight = height;

      if (pathTracer) {
        pathTracer->shutdown();
      }

      loadScene(ui_sceneFile);
      if (width != oldWidth || height != oldHeight) {
        resizeRenderTarget();
      }
      lastSceneFile = ui_sceneFile;
      iteration = 0;
    }
  
    if (scene == nullptr)
      return;

    if (lastLoopIterations != ui_iterations) {
      lastLoopIterations = ui_iterations;
      camchanged = true;
    }

    if (camchanged) {
        iteration = 0;
        RenderCamera &cam = scene->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        glm::vec3 v = -glm::normalize(cameraPosition);
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    if (iteration == 0) {
        pathTracer->shutdown();
        pathTracer->initialize();
        pathTracer->setScene(*scene);
        pathTracer->setCamera(scene->camera);
        pathTracer->setTraceDepth(scene->settings.traceDepth);
        pathTracer->resize(scene->camera.resolution.x, scene->camera.resolution.y);
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer
    cudaGraphicsMapResources(1, &pbo_resource, 0);
    uchar4* pbo_dptr = nullptr;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void**)&pbo_dptr, &num_bytes, pbo_resource);

    bool denoise = ui_denoise;
    if (iteration < ui_iterations) {
      iteration++;

      // execute the kernel
      int frame = 0;
      denoise = ui_denoise && iteration == ui_iterations;  // only denoise after the last iteration
      pathTracer->setCamera(scene->camera);
      pathTracer->setTraceDepth(scene->settings.traceDepth);
      pathTracer->renderSample(denoise);
    }

    if (ui_showGbuffer) {
      pathTracer->copyGBufferToPbo(pbo_dptr, ui_GbufferMode);
    }
    else {
      pathTracer->copyImageToPbo(pbo_dptr, denoise);
    }

    // unmap buffer object
    cudaGraphicsUnmapResources(1, &pbo_resource, 0);

    if (ui_saveAndExit) {
      saveImage();
      pathTracer->shutdown();
      cudaDeviceReset();
      exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (action == GLFW_PRESS) {
      switch (key) {
      case GLFW_KEY_1:
        ui_GbufferMode = GBUFFER_NORMAL;
        break;
      case GLFW_KEY_2:
        ui_GbufferMode = GBUFFER_POSITION;
        break;
      case GLFW_KEY_3:
        ui_GbufferMode = GBUFFER_TIME;
        break;
      case GLFW_KEY_ESCAPE:
        saveImage();
        glfwSetWindowShouldClose(window, GL_TRUE);
        break;
      case GLFW_KEY_S:
        saveImage();
        break;
      case GLFW_KEY_SPACE:
        camchanged = true;
        RenderCamera &cam = scene->camera;
        cam.lookAt = ogLookAt;
        break;
      }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
  rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
  middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
  if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
  if (leftMousePressed) {
    // compute new camera parameters
    phi -= (xpos - lastX) / width;
    theta -= (ypos - lastY) / height;
    theta = std::fmax(0.001f, std::fmin(theta, PI));
    camchanged = true;
  }
  else if (rightMousePressed) {
    zoom += (ypos - lastY) / height;
    zoom = std::fmax(0.1f, zoom);
    camchanged = true;
  }
  else if (middleMousePressed) {
    RenderCamera &cam = scene->camera;
    CameraBasis basis = computeCameraBasis(cam);
    glm::vec3 forward = basis.view;
    forward.y = 0.0f;
    forward = glm::normalize(forward);
    glm::vec3 right = basis.right;
    right.y = 0.0f;
    right = glm::normalize(right);

    cam.lookAt -= (float) (xpos - lastX) * right * 0.01f;
    cam.lookAt += (float) (ypos - lastY) * forward * 0.01f;
    camchanged = true;
  }
  lastX = xpos;
  lastY = ypos;
}
