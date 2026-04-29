#pragma once

#include <optix.h>

#include <sstream>
#include <stdexcept>

inline void optixCheck(OptixResult result, const char* call, const char* file, int line) {
    if (result == OPTIX_SUCCESS) {
        return;
    }

    std::ostringstream message;
    message << "OptiX call '" << call << "' failed at " << file << ':' << line
            << " with " << optixGetErrorName(result) << ": "
            << optixGetErrorString(result);
    throw std::runtime_error(message.str());
}

#define OPTIX_CHECK(call) optixCheck((call), #call, __FILE__, __LINE__)
