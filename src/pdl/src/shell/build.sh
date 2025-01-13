#!/bin/bash
set -e

USE_FULL_API=TRUE
TARGET_OS=linux
TARGET_ABI=amd64

function readlinkf() {
  perl -MCwd -e 'print Cwd::abs_path shift' "$1";
}

# 简化cmake参数
CMAKE_COMMAND_ARGS="-DCMAKE_VERBOSE_MAKEFILE=ON \
                   -DUSE_FULL_API=${USE_FULL_API} \
                   -DTARGET_OS=${TARGET_OS} \
                   -DTARGET_ABI=${TARGET_ABI} \
                   -DPADDLE_LITE_DIR=$(readlinkf ../../libs/PaddleLite) \
                   -DOpenCV_DIR=$(readlinkf ../../libs/OpenCV)"

BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}

rm -rf $BUILD_DIR
mkdir $BUILD_DIR
cd $BUILD_DIR
cmake ${CMAKE_COMMAND_ARGS} ..
make