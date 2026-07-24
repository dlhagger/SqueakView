cmake_minimum_required(VERSION 3.16)

project(flir_gst_source LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(GST REQUIRED IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-base-1.0
  gstreamer-video-1.0
)

add_library(gstflirspinsrc SHARED
  src/gstflirspinsrc.cpp
)

target_include_directories(gstflirspinsrc SYSTEM PRIVATE
  /opt/spinnaker/include
  /opt/nvidia/deepstream/deepstream/sources/includes
)

target_link_directories(gstflirspinsrc PRIVATE
  /opt/spinnaker/lib
  /opt/nvidia/deepstream/deepstream/lib
)

target_link_libraries(gstflirspinsrc PRIVATE
  PkgConfig::GST
  Spinnaker
  nvds_meta
  nvdsgst_meta
)

target_compile_definitions(gstflirspinsrc PRIVATE
  PACKAGE="squeakview"
)

target_compile_options(gstflirspinsrc PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

set_target_properties(gstflirspinsrc PROPERTIES
  PREFIX ""
  OUTPUT_NAME "gstflirspinsrc"
  BUILD_RPATH "/opt/spinnaker/lib;/opt/nvidia/deepstream/deepstream/lib"
  INSTALL_RPATH "/opt/spinnaker/lib;/opt/nvidia/deepstream/deepstream/lib"
)
