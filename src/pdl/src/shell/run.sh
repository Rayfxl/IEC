#!/bin/bash

# 模型配置
MODEL_NAME=mobilenet_v1_fp32_224
if [ -n "$1" ]; then
  MODEL_NAME=$1
fi

# 下载模型
if [ ! -d "../assets/models/$MODEL_NAME" ];then
  MODEL_URL="http://paddlelite-demo.bj.bcebos.com/devices/generic/models/${MODEL_NAME}.tar.gz"
  echo "Model $MODEL_NAME not found! Try to download it from $MODEL_URL ..."
  curl $MODEL_URL -o -| tar -xz -C ../assets/models
  if [[ $? -ne 0 ]]; then
    echo "Model $MODEL_NAME download failed!"
    exit 1
  fi
fi

# 固定配置(x86 CPU推理)
CONFIG_NAME=imagenet_224.txt
DATASET_NAME=test
TARGET_OS=linux
TARGET_ABI=amd64
NNADAPTER_DEVICE_NAMES="cpu"
NNADAPTER_CONTEXT_PROPERTIES="null"

# 设置环境变量
export GLOG_v=5
export LD_LIBRARY_PATH=../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib/cpu:../../libs/PaddleLite/$TARGET_OS/$TARGET_ABI/lib:.:$LD_LIBRARY_PATH

# 运行推理
BUILD_DIR=build.${TARGET_OS}.${TARGET_ABI}
chmod +x ./$BUILD_DIR/resnet18_pdl
./$BUILD_DIR/resnet18_pdl \
  ../assets/models/$MODEL_NAME \
  ../assets/configs/$CONFIG_NAME \
  ../assets/datasets/$DATASET_NAME \
  $NNADAPTER_DEVICE_NAMES \
  $NNADAPTER_CONTEXT_PROPERTIES \
  null null null null