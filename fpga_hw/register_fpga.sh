#!/bin/sh

BIN_NAME=productFpga
BUCKET=outlandish-watch-fpga

cd `dirname $0`

rm -rf ./to_aws

$SDACCEL_DIR/tools/create_sdaccel_afi.sh \
  -xclbin=../examples/${BIN_NAME}.xclbin \
  -o=${BIN_NAME} \
  -s3_bucket=${BUCKET} \
  -s3_dcp_key=dcp \
  -s3_logs_key=log

aws s3 cp ${BIN_NAME}.awsxclbin s3://${BUCKET}/${BIN_NAME}.awsxclbin

