project(InferenceEngineContrast)
cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# hard code, need change to you ncnn install dir
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/home/ray/ncnn/build/install")
find_package(OpenCV REQUIRED)
find_package(ncnn)
if(ncnn_FOUND)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/ncnn)

    add_executable(resnet18_ncnn src/ncnn/resnet18.cpp)
    target_link_libraries(resnet18_ncnn ncnn ${OpenCV_LIBS})

else()
    message(WARNING "not find ncnn ,please check CMAKE_PREFIX_PATH ,and change to you ncnn install dir")
endif()   


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin/mnn)
# hard code, need change to you MNN install dir
set(MNN_INCLUDE_DIR "/usr/local/include")
set(MNN_LIBRARY "/usr/local/lib/libMNN.so")
add_executable(resnet18_mnn src/mnn/resnet18.cpp)
target_link_libraries(resnet18_mnn ${MNN_LIBRARY} ${OpenCV_LIBS}) 

# 添加PDL编译目标
add_custom_target(pdl
    COMMAND cd ${CMAKE_SOURCE_DIR}/src/pdl/src/shell && ./build.sh linux amd64
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building PDL project..."
)



