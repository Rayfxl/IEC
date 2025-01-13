cmake_minimum_required(VERSION 3.10)

project(demo)

# System settings
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_CXX_COMPILER "g++")

# Paths
message(STATUS "PADDLE LITE DIR: ${PADDLE_LITE_DIR}")
message(STATUS "OpenCV DIR: ${OpenCV_DIR}")

# Include & Link dirs
include_directories(${PADDLE_LITE_DIR}/linux/amd64/include)
link_directories(${PADDLE_LITE_DIR}/linux/amd64/lib)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Find OpenCV
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Build target
add_executable(resnet18_pdl resnet18.cpp)

# Import PaddleLite lib
add_library(paddle_api_shared SHARED IMPORTED)
set_target_properties(paddle_api_shared PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LITE_DIR}/linux/amd64/lib/libpaddle_full_api_shared.so)

# Link
target_link_libraries(resnet18_pdl paddle_api_shared ${OpenCV_LIBS})