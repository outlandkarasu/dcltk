#!/bin/sh

cd `dirname $0`

TARGET=sw_emu
HOST_EXE=dcltk

cd ../
XCL_EMULATION_MODE=${TARGET} ./${HOST_EXE}

