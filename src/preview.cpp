#define _CRT_SECURE_NO_DEPRECATE
#include <ctime>
#include "main.h"
#include "preview.h"
#include <chrono>
#include <iostream>

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"
#include "ImGuiFileDialog/ImGuiFileDialog.h"

GLuint positionLocation = 0;
GLuint texcoordsLocation = 1;
GLuint pbo;
GLuint displayImage;

GLFWwindow *window;

std::string currentTimeString() {
    time_t now;
    time(&now);
    char buf[sizeof "0000-00-00_00-00-00z"];
    strftime(buf, sizeof buf, "%Y-%m-%d_%H-%M-%Sz", gmtime(&now));
    return std::string(buf);
}

//-------------------------------
//----------SETUP STUFF----------
//-------------------------------

void initTextures() {
    glGenTextures(1, &displayImage);
    glBindTexture(GL_TEXTURE_2D, displayImage);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);
}

void initVAO(void) {
    GLfloat vertices[] = {
        -1.0f, -1.0f,
        1.0f, -1.0f,
        1.0f,  1.0f,
        -1.0f,  1.0f,
    };

    GLfloat texcoords[] = {
        1.0f, 1.0f,
        0.0f, 1.0f,
        0.0f, 0.0f,
        1.0f, 0.0f
    };

    GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

    GLuint vertexBufferObjID[3];
    glGenBuffers(3, vertexBufferObjID);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(positionLocation);

    glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
    glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(texcoordsLocation);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
}

GLuint initShader() {
    const char *attribLocations[] = { "Position", "Texcoords" };
    GLuint program = glslUtility::createDefaultProgram(attribLocations, 2);
    GLint location;

    //glUseProgram(program);
    if ((location = glGetUniformLocation(program, "u_image")) != -1) {
        glUniform1i(location, 0);
    }

    return program;
}

void deletePBO(GLuint* pbo) {
    if (pbo) {
        // unregister this buffer object with CUDA
        cudaGraphicsUnregisterResource(pbo_resource);

        glBindBuffer(GL_ARRAY_BUFFER, *pbo);
        glDeleteBuffers(1, pbo);

        *pbo = (GLuint)NULL;
    }
}

void deleteTexture(GLuint* tex) {
    glDeleteTextures(1, tex);
    *tex = (GLuint)NULL;
}

void cleanupCuda() {
    if (pbo) {
        deletePBO(&pbo);
    }
    if (displayImage) {
        deleteTexture(&displayImage);
    }
}

void initCuda() {

    // Clean up on program exit
    atexit(cleanupCuda);
}

void initPBO() {
    // set up vertex data parameter
    int num_texels = width * height;
    int num_values = num_texels * 4;
    int size_tex_data = sizeof(GLubyte) * num_values;

    // Generate a buffer ID called a PBO (Pixel Buffer Object)
    glGenBuffers(1, &pbo);

    // Make this the current UNPACK buffer (OpenGL is state-based)
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

    // Allocate data for the buffer. 4-channel 8-bit image
    glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);

    cudaGraphicsGLRegisterBuffer(&pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

void errorCallback(int error, const char* description) {
    fprintf(stderr, "%s\n", description);
}

bool init() {
    glfwSetErrorCallback(errorCallback);

    if (!glfwInit()) {
        exit(EXIT_FAILURE);
    }

    window = glfwCreateWindow(width, height, "CIS 565 Path Tracer", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetCursorPosCallback(window, mousePositionCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);

    // Set up GL context
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        return false;
    }

    // Initialize other stuff
    initVAO();
    initTextures();
    initCuda();
    initPBO();
    GLuint passthroughProgram = initShader();

    glUseProgram(passthroughProgram);
    glActiveTexture(GL_TEXTURE0);

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    //// Setup Dear ImGui style
    ImGui::StyleColorsDark();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    return true;
}

using time_point_t = std::chrono::high_resolution_clock::time_point;

static ImGuiWindowFlags windowFlags = ImGuiWindowFlags_None | ImGuiWindowFlags_NoMove;
static bool ui_hide = false;

void drawGui(int windowWidth, int windowHeight) {
    // Dear imgui new frame
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Dear imgui define
    ImVec2 minSize(300.f, 100.f);
    ImVec2 maxSize((float)windowWidth * 0.5, (float)windowHeight * 0.3);
    ImGui::SetNextWindowSizeConstraints(minSize, maxSize);

    ImGui::SetNextWindowPos(ui_hide ? ImVec2(-1000.f, -1000.f) : ImVec2(0.0f, 0.0f));

    ImGui::Begin("Control Panel", 0, windowFlags);
    ImGui::SetWindowFontScale(1);

    ImGui::Text("press H to hide GUI completely.");
    if (ImGui::IsKeyPressed(ImGuiKey::ImGuiKey_H)) {
      ui_hide = !ui_hide;
    }

    if (ImGui::CollapsingHeader("Render Settings")) {
      // open Dialog Simple
      if (ImGui::Button("Change Scene"))
        ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose Scene File", ".txt", "../scenes/.");

      // display
      if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
      {
        // action if OK
        if (ImGuiFileDialog::Instance()->IsOk())
        {
          ui_sceneFile = ImGuiFileDialog::Instance()->GetFilePathName();
          //std::string filePath = ImGuiFileDialog::Instance()->GetCurrentPath();
        }

        // close
        ImGuiFileDialog::Instance()->Close();
      }

      ImGui::SliderInt("Iterations", &ui_iterations, 1, startupIterations);
    }

    if (ImGui::CollapsingHeader("Denoise Settings")) {
      ImGui::Checkbox("Enable", &ui_denoise);

      ImGui::SliderInt("Filter Passes", &ui_filterPasses, 1, 10);
      ImGui::SliderInt("Filter Size", &ui_filterSize, 0, 100);
      ImGui::SliderFloat("Color Weight", &ui_colorWeight, 0.0f, 5.0f);
      ImGui::SliderFloat("Normal Weight", &ui_normalWeight, 0.0f, 5.0f);
      ImGui::SliderFloat("Position Weight", &ui_positionWeight, 0.0f, 5.0f);

      ImGui::Separator();

      ImGui::Checkbox("Show GBuffer", &ui_showGbuffer);

      ImGui::Separator();
    }

    if (ImGui::Button("Save image and exit")) {
      ui_saveAndExit = true;
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void mainLoop() {
  
    //time_point_t time_start_cpu;
    //time_point_t time_end_cpu;
    //time_start_cpu = std::chrono::high_resolution_clock::now();
    while (window && !glfwWindowShouldClose(window)) {
        glfwPollEvents();
        runCuda();

        string title = "CIS565 Path Tracer | " + utilityCore::convertIntToString(iteration) + " Iterations";
        glfwSetWindowTitle(window, title.c_str());

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
        glBindTexture(GL_TEXTURE_2D, displayImage);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
        glClear(GL_COLOR_BUFFER_BIT);

        // VAO, shader program, and texture already bound
        glDrawElements(GL_TRIANGLES, 6,  GL_UNSIGNED_SHORT, 0);

        // Draw imgui
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        drawGui(display_w, display_h);

        glfwSwapBuffers(window);
        //time_end_cpu = std::chrono::high_resolution_clock::now();
        //std::chrono::duration<double, std::milli> duro = time_end_cpu - time_start_cpu;
        //float duration = static_cast<float>(duro.count());
        //std::cout << iteration << " " << duration << endl;
    }
    glfwDestroyWindow(window);
    glfwTerminate();
}
