cmake_minimum_required(VERSION 3.1)

project(cis565_path_tracer)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# CUDA 13.x's bundled CCCL/Thrust requires at least C++17.
set(CMAKE_CXX_STANDARD 17)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    # Set the possible values of build type for cmake-gui
    SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

########################################
# CUDA Setup
########################################
if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 4)
    message(FATAL_ERROR
        "This project must be configured for 64-bit on Windows. "
        "The current generator platform is Win32, but the installed CUDA toolkit only provides "
        "64-bit runtime libraries such as lib/x64/cudart.lib. "
        "In VS Code, select an x64 Visual Studio kit/generator or reconfigure with '-A x64'.")
endif()

find_package(CUDA 10 REQUIRED)
include(${CMAKE_MODULE_PATH}/CUDAComputesList.cmake)
find_package(OptiX REQUIRED)

set(CUDA_CCCL_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include/cccl")
include_directories(${CUDA_TOOLKIT_INCLUDE})
if(EXISTS "${CUDA_CCCL_INCLUDE_DIR}")
    include_directories(${CUDA_CCCL_INCLUDE_DIR})
endif()

list(APPEND CUDA_NVCC_FLAGS ${CUDA_GENERATE_CODE})
list(APPEND CUDA_NVCC_FLAGS_DEBUG "-g -G")
list(APPEND CUDA_NVCC_FLAGS "--std=c++17")
set(CUDA_VERBOSE_BUILD ON)

if(WIN32)
    # Set up include and lib paths
    add_definitions(-DNOMINMAX -DWIN32_LEAN_AND_MEAN)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zc:preprocessor")
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "Host side compiler used by NVCC" FORCE)
    include_directories(external/include/ImGuiFileDialog/dirent)
endif(WIN32)
########################################

find_package(OpenGL REQUIRED)

if(UNIX)
    find_package(glfw3 REQUIRED)
    find_package(GLEW REQUIRED)
    set(LIBRARIES glfw ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY})
else(UNIX)
    set(EXTERNAL "external")

    set(GLFW_ROOT_DIR ${EXTERNAL})
    set(GLFW_USE_STATIC_LIBS ON)
    find_package(GLFW REQUIRED)

    set(GLEW_ROOT_DIR ${EXTERNAL})
    set(GLEW_USE_STATIC_LIBS ON)
    find_package(GLEW REQUIRED)

    add_definitions(${GLEW_DEFINITIONS})
    include_directories(${GLEW_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
    set(LIBRARIES ${GLEW_LIBRARY} ${GLFW_LIBRARY} ${OPENGL_LIBRARY})
endif(UNIX)

set(GLM_ROOT_DIR "external")
find_package(GLM REQUIRED)
include_directories(${GLM_INCLUDE_DIRS})
include_directories(${OptiX_INCLUDE_DIRS})
include_directories(
    imgui
    src
    src/app
    src/geometry
    src/renderer
    src/renderer/denoise
    src/renderer/gpu
    src/scene
    )

set(headers
    src/app/main.h
    src/app/preview.h
    src/core/glslUtility.hpp
    src/core/image.h
    src/core/math.h
    src/core/types.h
    src/core/utilities.h
    src/geometry/interactions.h
    src/geometry/intersections.h
    src/geometry/octree.h
    src/renderer/PathTracer.h
    src/renderer/PathTracerKernels.cuh
    src/renderer/RenderSettings.h
    src/renderer/RenderState.h
    src/renderer/denoise/OptiXDenoiser.h
    src/renderer/denoise/OptiXDenoiserUtil.h
    src/renderer/denoise/optixCheck.h
    src/renderer/gpu/CudaBuffer.h
    src/renderer/gpu/CudaCheck.h
    src/renderer/gpu/CudaTexture2D.h
    src/renderer/gpu/GpuFramebuffer.h
    src/renderer/gpu/GpuScene.h
    src/scene/GpuSceneTypes.h
    src/scene/GltfSceneLoader.h
    src/scene/RenderScene.h
    src/scene/SceneLoader.h
    src/scene/TextSceneLoader.h
    )

set(sources
    src/app/main.cpp
    src/app/preview.cpp
    src/core/glslUtility.cpp
    src/core/image.cpp
    src/core/stb.cpp
    src/core/utilities.cpp
    src/geometry/octree.cpp
    src/renderer/PathTracer.cpp
    src/renderer/PathTracerKernels.cu
    src/renderer/gpu/GpuFramebuffer.cu
    src/renderer/gpu/GpuScene.cu
    src/scene/GltfSceneLoader.cpp
    src/scene/TextSceneLoader.cpp
    )

set(imgui
    imgui/imconfig.h
    imgui/imgui.cpp
    imgui/imgui.h
    imgui/imgui_draw.cpp
    imgui/imgui_internal.h
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_impl_glfw.cpp
    imgui/imgui_impl_glfw.h
    imgui/imgui_impl_opengl2.cpp
    imgui/imgui_impl_opengl2.h
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_impl_opengl3.h
    imgui/imgui_impl_opengl3_loader.h
    imgui/imstb_rectpack.h
    imgui/imstb_textedit.h
    imgui/imstb_truetype.h
    )

set(imguifile external/include/ImGuiFileDialog/ImGuiFileDialog.cpp)

list(SORT headers)
list(SORT sources)
list(SORT imgui)

source_group(Headers FILES ${headers})
source_group(Sources FILES ${sources})
source_group(imgui FILES ${imgui})

#add_subdirectory(stream_compaction)  # TODO: uncomment if using your stream compaction

cuda_add_executable(${CMAKE_PROJECT_NAME} ${sources} ${headers} ${imgui} ${imguifile})
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${OptiX_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME}
    ${LIBRARIES}
    #stream_compaction  # TODO: uncomment if using your stream compaction
    )
