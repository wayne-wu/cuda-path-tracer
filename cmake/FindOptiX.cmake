include(FindPackageHandleStandardArgs)

set(_OptiX_HINTS)

foreach(_OptiX_VAR OPTIX_ROOT_DIR OPTIX_SDK_ROOT OptiX_ROOT_DIR OptiX_SDK_ROOT)
    if(DEFINED ${_OptiX_VAR})
        list(APPEND _OptiX_HINTS "${${_OptiX_VAR}}")
    endif()
    if(DEFINED ENV{${_OptiX_VAR}})
        list(APPEND _OptiX_HINTS "$ENV{${_OptiX_VAR}}")
    endif()
endforeach()

if(WIN32)
    file(GLOB _OptiX_WINDOWS_INSTALLS
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK *"
        "C:/Program Files/NVIDIA Corporation/OptiX SDK *")
    if(_OptiX_WINDOWS_INSTALLS)
        list(SORT _OptiX_WINDOWS_INSTALLS)
        list(REVERSE _OptiX_WINDOWS_INSTALLS)
        list(APPEND _OptiX_HINTS ${_OptiX_WINDOWS_INSTALLS})
    endif()
endif()

find_path(OptiX_INCLUDE_DIR
    NAMES optix.h optix_stubs.h optix_device.h
    HINTS ${_OptiX_HINTS}
    PATH_SUFFIXES include
    DOC "Path to the NVIDIA OptiX SDK include directory")

find_package_handle_standard_args(OptiX
    REQUIRED_VARS OptiX_INCLUDE_DIR)

if(OptiX_FOUND)
    set(OptiX_INCLUDE_DIRS ${OptiX_INCLUDE_DIR})
    if(NOT TARGET OptiX::OptiX)
        add_library(OptiX::OptiX INTERFACE IMPORTED)
        set_target_properties(OptiX::OptiX PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${OptiX_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(OptiX_INCLUDE_DIR)
