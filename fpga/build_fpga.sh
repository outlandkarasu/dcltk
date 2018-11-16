#!/bin/sh

set -e

cd `dirname $0`

#TARGET=sw_emu
TARGET=hw_emu
KERNEL_NAME=product
PLATFORM=${AWS_PLATFORM}
SOURCE=../examples/productFpga.cl
OBJECT_FILE=../examples/productFpga.xo
XCLBIN_FILE=../examples/productFpga.xclbin
REPORT_DIR=../fpga_reports

mkdir -p ${REPORT_DIR}

xocc -c \
  -k ${KERNEL_NAME} \
  --target ${TARGET} \
  --platform ${PLATFORM} \
  ${SOURCE} \
  -o ${OBJECT_FILE} \
  --save-temps

xocc -l \
  --nk ${KERNEL_NAME}:1 \
  --target ${TARGET} \
  --platform ${PLATFORM} \
  ${OBJECT_FILE} \
  -o ${XCLBIN_FILE} \
  --save-temps \
  --profile_kernel data:all:all:all:all \
  --report_level estimate --report_dir ${REPORT_DIR}

emconfigutil \
  --platform ${PLATFORM} \
  --nd 1 \
  --od ../

#./register_fpga.sh

