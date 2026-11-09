################################################################################
# Copyright (c) 2018-2024, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# Edited by Marcos Luciano
# https://www.github.com/marcoslucianops
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
$(error "CUDA_VER is not set")
endif

OPENCV?=
ifeq ($(OPENCV),)
OPENCV=0
endif

GRAPH?=
ifeq ($(GRAPH),)
GRAPH=0
endif

DEEPSTREAM_SDK?=$(if $(SQUEAKVIEW_DEEPSTREAM_SDK),$(SQUEAKVIEW_DEEPSTREAM_SDK),/opt/nvidia/deepstream/deepstream)
DEEPSTREAM_INCLUDES?=$(patsubst %/nvdsinfer_custom_impl.h,%,$(firstword $(wildcard $(DEEPSTREAM_SDK)/includes/nvdsinfer_custom_impl.h $(DEEPSTREAM_SDK)/sources/includes/nvdsinfer_custom_impl.h)))
ifeq ($(DEEPSTREAM_INCLUDES),)
$(error "DeepStream headers missing under $(DEEPSTREAM_SDK)")
endif

CC:= g++
NVCC:=/usr/local/cuda-$(CUDA_VER)/bin/nvcc

CFLAGS:= -Wall -std=c++11 -shared -fPIC -Wno-error=deprecated-declarations
CFLAGS+= -I$(DEEPSTREAM_INCLUDES) -I/usr/local/cuda-$(CUDA_VER)/include

ifeq ($(OPENCV), 1)
COMMON+= -DOPENCV
CFLAGS+= $(shell pkg-config --cflags opencv4 2> /dev/null || pkg-config --cflags opencv)
LIBS+= $(shell pkg-config --libs opencv4 2> /dev/null || pkg-config --libs opencv)
endif

ifeq ($(GRAPH), 1)
COMMON+= -DGRAPH
endif

CUFLAGS:= -I$(DEEPSTREAM_INCLUDES) -I/usr/local/cuda-$(CUDA_VER)/include

ifeq ($(shell ldconfig -p | grep -q libnvparsers && echo 1 || echo 0), 1)
LIBS+= -lnvparsers
endif

LIBS+= -lnvinfer_plugin -lnvinfer -lnvonnxparser -L/usr/local/cuda-$(CUDA_VER)/lib64 -lcudart -lcublas -lstdc++fs
LFLAGS:= -shared -Wl,--start-group $(LIBS) -Wl,--end-group

INCS:= $(wildcard layers/*.h)
INCS+= $(wildcard *.h)

SRCFILES:= $(filter-out calibrator.cpp, $(wildcard *.cpp))

ifeq ($(OPENCV), 1)
SRCFILES+= calibrator.cpp
endif

SRCFILES+= $(wildcard layers/*.cpp)
SRCFILES+= $(wildcard *.cu)

TARGET_LIB:= libnvdsinfer_custom_impl_Yolo.so

TARGET_OBJS:= $(SRCFILES:.cpp=.o)
TARGET_OBJS:= $(TARGET_OBJS:.cu=.o)

all: $(TARGET_LIB)

# Make does not track command-line flag changes. Refresh this stamp only when
# SDK identity, headers, CUDA selection, or compilation/link flags change.
BUILD_CONFIG:= .build-config
.PHONY: all clean FORCE
FORCE:
$(BUILD_CONFIG): FORCE
	@set -e; { printf '%s\n' "$(CC)" "$(NVCC)" "$(CFLAGS)" "$(CUFLAGS)" "$(COMMON)" "$(LFLAGS)"; \
	  readlink -f "$(DEEPSTREAM_SDK)"; \
	  cat "$(DEEPSTREAM_SDK)/version"; \
	  sha256sum "$(DEEPSTREAM_INCLUDES)/nvdsinfer_custom_impl.h" "$(DEEPSTREAM_INCLUDES)/nvdsinfer.h"; \
	} > $@.tmp
	@cmp -s $@.tmp $@ && rm -f $@.tmp || mv -f $@.tmp $@

%.o: %.cpp $(INCS) Makefile $(BUILD_CONFIG)
	$(CC) -c $(COMMON) -o $@ $(CFLAGS) $<

%.o: %.cu $(INCS) Makefile $(BUILD_CONFIG)
	$(NVCC) -c -o $@ --compiler-options '-fPIC' $(CUFLAGS) $<

$(TARGET_LIB) : $(TARGET_OBJS) $(BUILD_CONFIG)
	$(CC) -o $@  $(TARGET_OBJS) $(LFLAGS)

clean:
	rm -rf $(TARGET_LIB)
	rm -rf $(TARGET_OBJS)
	rm -f $(BUILD_CONFIG) $(BUILD_CONFIG).tmp
