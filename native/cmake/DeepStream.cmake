# Native builds target DeepStream 9.2 only.
set(_deepstream_default "/opt/nvidia/deepstream/deepstream")
if(DEFINED ENV{SQUEAKVIEW_DEEPSTREAM_SDK} AND NOT "$ENV{SQUEAKVIEW_DEEPSTREAM_SDK}" STREQUAL "")
  set(_deepstream_default "$ENV{SQUEAKVIEW_DEEPSTREAM_SDK}")
endif()
set(DEEPSTREAM_SDK "${_deepstream_default}" CACHE PATH "DeepStream SDK root")
# Enforce this for direct CMake builds as well as the shell entry points.
if(NOT EXISTS "${DEEPSTREAM_SDK}/version")
  message(FATAL_ERROR "Cannot read DeepStream SDK version: ${DEEPSTREAM_SDK}/version")
endif()
file(READ "${DEEPSTREAM_SDK}/version" _deepstream_version)
if(NOT _deepstream_version MATCHES "(^|\n)Version: +9[.]2[.][0-9]+([^0-9A-Za-z_]|$)")
  message(FATAL_ERROR "SqueakView requires DeepStream 9.2: ${DEEPSTREAM_SDK}")
endif()
# Headers always follow the selected SDK, including on cached reruns.
if(NOT EXISTS "${DEEPSTREAM_SDK}/includes/nvdsinfer_custom_impl.h")
  message(FATAL_ERROR "DeepStream headers missing under ${DEEPSTREAM_SDK}/includes")
endif()
set(DEEPSTREAM_INCLUDES "${DEEPSTREAM_SDK}/includes" CACHE PATH "Headers for the selected DeepStream SDK" FORCE)
