#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

inline void cudaCheck(cudaError_t result, const char* expression, const char* file, int line) {
    if (result == cudaSuccess) {
        return;
    }

    throw std::runtime_error(
        std::string("CUDA error at ") + file + ":" + std::to_string(line) +
        " for " + expression + ": " + cudaGetErrorString(result));
}

#define CUDA_CHECK(expression) cudaCheck((expression), #expression, __FILE__, __LINE__)

