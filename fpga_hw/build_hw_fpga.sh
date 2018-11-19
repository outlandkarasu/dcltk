#!/bin/sh

set -e

cd `dirname $0`

#TARGET=sw_emu
TARGET=hw
KERNEL_NAME=product
KERNEL_COUNT=16
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
  --max_memory_ports ${KERNEL_NAME} \
  --save-temps \
  -o ${OBJECT_FILE} \
  ${SOURCE}

xocc -l \
  --target ${TARGET} \
  --platform ${PLATFORM} \
  --max_memory_ports ${KERNEL_NAME} \
  --sp ${KERNEL_NAME}_1.m_axi_gmem0:bank0 \
  --sp ${KERNEL_NAME}_1.m_axi_gmem1:bank1 \
  --sp ${KERNEL_NAME}_1.m_axi_gmem2:bank2 \
  --nk ${KERNEL_NAME}:${KERNEL_COUNT} \
  --save-temps \
  --profile_kernel data:all:all:all:all \
  --report_level estimate --report_dir ${REPORT_DIR} \
  -o ${XCLBIN_FILE} \
  ${OBJECT_FILE}

emconfigutil \
  --platform ${PLATFORM} \
  --nd 1 \
  --od ../

./register_fpga.sh

