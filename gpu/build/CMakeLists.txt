cmake_minimum_required(VERSION 3.10)
project(kinfu_opencl CXX C)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------
# Paths – can be overridden via cmake -D flags (e.g. for Docker builds)
#
#   cmake . -DCONDA_CELLSEG1=/usr -DCONDA_GLUE=/usr \
#           -DOpenCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libOpenCL.so
# -----------------------------------------------------------------------
set(CONDA_CELLSEG1 "/home/jiy/miniconda3/envs/cellseg1"
    CACHE PATH "Path to cellseg1 conda env (provides Boost + Eigen3)")
set(CONDA_GLUE "/home/jiy/miniconda3/envs/glue_env"
    CACHE PATH "Path to glue_env conda env (provides OpenCL headers)")

# Stub / generated headers live here (direct.h, pcl stubs, gpu containers)
set(STUB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Source tree root
set(GPU_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

# -----------------------------------------------------------------------
# Find OpenCL
# -----------------------------------------------------------------------
# Header from glue_env (or override), lib from system (or override)
set(OpenCL_INCLUDE_DIR "${CONDA_GLUE}/include")
set(OpenCL_LIBRARY "/usr/lib/x86_64-linux-gnu/libOpenCL.so.1.0.0"
    CACHE FILEPATH "Path to libOpenCL shared library")

if(NOT EXISTS "${OpenCL_INCLUDE_DIR}/CL/cl.h")
    message(FATAL_ERROR "OpenCL header not found at ${OpenCL_INCLUDE_DIR}/CL/cl.h")
endif()

# -----------------------------------------------------------------------
# Boost (header-only parts we need: shared_ptr, make_shared)
# -----------------------------------------------------------------------
set(Boost_INCLUDE_DIR "${CONDA_CELLSEG1}/include")
set(Boost_LIBRARY_DIR "${CONDA_CELLSEG1}/lib")

# -----------------------------------------------------------------------
# Eigen3
# -----------------------------------------------------------------------
set(Eigen3_INCLUDE_DIR "${CONDA_CELLSEG1}/include/eigen3")

# -----------------------------------------------------------------------
# Collect sources (library only, no tools/test)
# -----------------------------------------------------------------------
file(GLOB LIB_SRCS "${GPU_SRC_DIR}/src/*.cpp" "${GPU_SRC_DIR}/src/*.c")
# Exclude temp/backup files
list(FILTER LIB_SRCS EXCLUDE REGEX ".*\\.TMP$")
list(FILTER LIB_SRCS EXCLUDE REGEX ".*~.*")
list(FILTER LIB_SRCS EXCLUDE REGEX ".*\\.bak$")
# Exclude Korean-named files (can't be reliably compiled by filename)
file(GLOB JUNK_FILES "${GPU_SRC_DIR}/src/*.cpp~*" "${GPU_SRC_DIR}/src/* - *")
foreach(f ${JUNK_FILES})
    list(REMOVE_ITEM LIB_SRCS "${f}")
endforeach()

# Remove any non-ASCII named files explicitly
file(GLOB ALL_CPP "${GPU_SRC_DIR}/src/*.cpp")
foreach(f ${ALL_CPP})
    get_filename_component(fname "${f}" NAME)
    # Check for tilde in name (backup file pattern used by Windows)
    string(FIND "${fname}" "~" tilde_pos)
    if(NOT tilde_pos EQUAL -1)
        list(REMOVE_ITEM LIB_SRCS "${f}")
    endif()
endforeach()

message(STATUS "Library sources: ${LIB_SRCS}")

# -----------------------------------------------------------------------
# Include directories
# -----------------------------------------------------------------------
include_directories(
    # Stubs first (direct.h, pcl stubs, gpu containers, opencv stub)
    "${STUB_INCLUDE_DIR}"
    # Project headers
    "${GPU_SRC_DIR}/include"
    "${GPU_SRC_DIR}/src"
    # OpenCL
    "${OpenCL_INCLUDE_DIR}"
    # Boost
    "${Boost_INCLUDE_DIR}"
    # Eigen
    "${Eigen3_INCLUDE_DIR}"
)

# -----------------------------------------------------------------------
# Compiler flags
# -----------------------------------------------------------------------
# Path to OpenCL kernel sources — passed as a compile-time define so the
# hardcoded relative paths in bilateral.cpp, maps.cpp, etc. can be overridden.
set(KINFU_CL_DIR "${GPU_SRC_DIR}/kinfu_opencl/src/cl")

add_definitions(
    -DPCLAPI_EXPORTS         # mark PCL_EXPORTS as dllexport equivalent
    -DKINFU_CL_DIR="${KINFU_CL_DIR}"
)

# Suppress common warnings from this codebase
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

# -----------------------------------------------------------------------
# Build the shared library
# -----------------------------------------------------------------------
add_library(kinfu_opencl STATIC ${LIB_SRCS})

target_link_libraries(kinfu_opencl
    "${OpenCL_LIBRARY}"
    "-L${Boost_LIBRARY_DIR}"
    "-Wl,-rpath,${Boost_LIBRARY_DIR}"
)

set_target_properties(kinfu_opencl PROPERTIES
    OUTPUT_NAME "kinfu_opencl"
)

# -----------------------------------------------------------------------
# Install (optional)
# -----------------------------------------------------------------------
install(TARGETS kinfu_opencl
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY "${GPU_SRC_DIR}/include/" DESTINATION include)

# -----------------------------------------------------------------------
# Standalone runner: run_kinfu_oni
# (synthetic depth – no OpenNI2 required)
# -----------------------------------------------------------------------
add_executable(run_kinfu_oni "${CMAKE_CURRENT_SOURCE_DIR}/run_kinfu_oni.cpp")

target_include_directories(run_kinfu_oni PRIVATE
    "${STUB_INCLUDE_DIR}"
    "${GPU_SRC_DIR}/include"
    "${GPU_SRC_DIR}/src"
    "${OpenCL_INCLUDE_DIR}"
    "${Boost_INCLUDE_DIR}"
    "${Eigen3_INCLUDE_DIR}"
)

target_link_libraries(run_kinfu_oni
    kinfu_opencl
    "${OpenCL_LIBRARY}"
    "-L${Boost_LIBRARY_DIR}"
    "-Wl,-rpath,${Boost_LIBRARY_DIR}"
    "-lm"
)

target_compile_options(run_kinfu_oni PRIVATE -w)

# -----------------------------------------------------------------------
# TUM RGB-D runner: run_kinfu_tum
# Reads 16-bit depth PNGs from a TUM RGB-D dataset sequence.
# Requires libpng.
# -----------------------------------------------------------------------
find_package(PNG REQUIRED)

add_executable(run_kinfu_tum "${CMAKE_CURRENT_SOURCE_DIR}/run_kinfu_tum.cpp")

target_include_directories(run_kinfu_tum PRIVATE
    "${STUB_INCLUDE_DIR}"
    "${GPU_SRC_DIR}/include"
    "${GPU_SRC_DIR}/src"
    "${OpenCL_INCLUDE_DIR}"
    "${Boost_INCLUDE_DIR}"
    "${Eigen3_INCLUDE_DIR}"
    ${PNG_INCLUDE_DIRS}
)

target_link_libraries(run_kinfu_tum
    kinfu_opencl
    "${OpenCL_LIBRARY}"
    "-L${Boost_LIBRARY_DIR}"
    "-Wl,-rpath,${Boost_LIBRARY_DIR}"
    ${PNG_LIBRARIES}
    "-lm"
)

target_compile_options(run_kinfu_tum PRIVATE -w)
