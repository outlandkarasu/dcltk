#!/bin/sh

cd `dirname $0`

#TARGET=hw_emu
TARGET=sw_emu
HOST_EXE=dcltk

cd ../
XCL_EMULATION_MODE=${TARGET} ./${HOST_EXE}

