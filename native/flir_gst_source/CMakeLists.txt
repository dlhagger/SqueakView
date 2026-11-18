cmake_minimum_required(VERSION 3.16)

project(flir_gst_source LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/DeepStream.cmake")

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
  "${DEEPSTREAM_INCLUDES}"
)

target_link_directories(gstflirspinsrc PRIVATE
  /opt/spinnaker/lib
  ${DEEPSTREAM_SDK}/lib
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
  BUILD_RPATH "/opt/spinnaker/lib;${DEEPSTREAM_SDK}/lib"
  INSTALL_RPATH "/opt/spinnaker/lib;${DEEPSTREAM_SDK}/lib"
)

include(CTest)
if(BUILD_TESTING)
  add_executable(live_mailbox_test tests/test_live_mailbox.cpp)
  target_compile_options(live_mailbox_test PRIVATE -Wall -Wextra -Wpedantic -UNDEBUG)
  add_test(NAME live_mailbox_isolation COMMAND live_mailbox_test)
  set_tests_properties(live_mailbox_isolation PROPERTIES TIMEOUT 10)
  add_executable(flir_source_eos_test tests/test_graceful_eos.cpp)
  add_dependencies(flir_source_eos_test live_mailbox_test)
  target_include_directories(flir_source_eos_test SYSTEM PRIVATE
    /opt/spinnaker/include
    "${DEEPSTREAM_INCLUDES}"
  )
  target_link_directories(flir_source_eos_test PRIVATE
    /opt/spinnaker/lib
    ${DEEPSTREAM_SDK}/lib
  )
  target_link_libraries(flir_source_eos_test PRIVATE
    PkgConfig::GST Spinnaker nvds_meta nvdsgst_meta
  )
  target_compile_definitions(flir_source_eos_test PRIVATE PACKAGE="squeakview")
  target_compile_options(flir_source_eos_test PRIVATE -Wall -Wextra -Wpedantic)
  set_target_properties(flir_source_eos_test PROPERTIES
    BUILD_RPATH "/opt/spinnaker/lib;${DEEPSTREAM_SDK}/lib"
  )
  add_test(NAME flir_source_graceful_eos COMMAND flir_source_eos_test)
  set_tests_properties(flir_source_graceful_eos PROPERTIES TIMEOUT 20)
endif()
