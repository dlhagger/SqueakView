#include <gst/base/gstpushsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"
#include "gstnvdsmeta.h"
#include "nvds_latency_meta.h"
#include "nvdsmeta.h"

using Spinnaker::CameraList;
using Spinnaker::CameraPtr;
using Spinnaker::ChunkData;
using Spinnaker::ImagePtr;
using Spinnaker::ImageProcessor;
using Spinnaker::PixelFormat_Mono8;
using Spinnaker::System;
using Spinnaker::SystemPtr;
using Spinnaker::GenApi::CBooleanPtr;
using Spinnaker::GenApi::CCommandPtr;
using Spinnaker::GenApi::CEnumEntryPtr;
using Spinnaker::GenApi::CEnumerationPtr;
using Spinnaker::GenApi::CFloatPtr;
using Spinnaker::GenApi::CIntegerPtr;
using Spinnaker::GenApi::CStringPtr;
using Spinnaker::GenApi::INodeMap;
using Spinnaker::GenApi::IsAvailable;
using Spinnaker::GenApi::IsReadable;
using Spinnaker::GenApi::IsWritable;

namespace {

constexpr guint DEFAULT_WIDTH = 1440;
constexpr guint DEFAULT_HEIGHT = 1080;
constexpr guint DEFAULT_FPS = 30;
constexpr guint DEFAULT_TIMEOUT_MS = 1000;
constexpr guint DEFAULT_MAX_CONSECUTIVE_TIMEOUTS = 10;
constexpr const char* DEFAULT_METADATA_PROFILE = "scientific";
constexpr char FLIR_FRAME_META_DESCRIPTOR[] = "SQUEAKVIEW.FLIR.FRAME_META.v1";

enum {
  PROP_0,
  PROP_CAMERA_INDEX,
  PROP_WIDTH,
  PROP_HEIGHT,
  PROP_FPS,
  PROP_PIXEL_FORMAT,
  PROP_TRIGGER,
  PROP_TRIGGER_ACTIVATION,
  PROP_EXPOSURE_US,
  PROP_GAIN,
  PROP_TIMEOUT_MS,
  PROP_DROP_INCOMPLETE,
  PROP_BUFFER_HANDLING,
  PROP_STREAM_BUFFER_COUNT,
  PROP_CAMERA_SERIAL,
  PROP_METADATA_PROFILE,
  PROP_MAX_CONSECUTIVE_TIMEOUTS,
  PROP_CAPTURE_LOG_PATH,
  PROP_FRAME_MANIFEST_PATH,
  PROP_CAMERA_TELEMETRY_PATH,
  PROP_ERROR_LOG_PATH,
  PROP_CAMERA_RUNTIME_PATH,
  PROP_FAULT_AFTER_FRAMES,
  PROP_FAULT_KIND,
};

struct _GstFlirSpinSrc {
  GstPushSrc parent;

  guint camera_index;
  guint width;
  guint height;
  guint fps;
  gchar* pixel_format;
  gboolean trigger;
  gchar* trigger_activation;
  gdouble exposure_us;
  gdouble gain;
  guint timeout_ms;
  gboolean drop_incomplete;
  gchar* buffer_handling;
  guint stream_buffer_count;
  gchar* camera_serial;
  gchar* metadata_profile;
  guint max_consecutive_timeouts;
  gchar* capture_log_path;
  FILE* capture_log;
  gchar* frame_manifest_path;
  FILE* frame_manifest;
  gchar* camera_telemetry_path;
  FILE* camera_telemetry;
  gchar* error_log_path;
  FILE* error_log;
  gchar* camera_runtime_path;
  guint64 fault_after_frames;
  gchar* fault_kind;
  gboolean fault_injected;
  gboolean latency_reference_enabled;

  guint actual_width;
  guint actual_height;
  gdouble actual_fps;
  gdouble actual_exposure_us;
  gdouble actual_gain_db;
  guint64 frame_count;
  guint64 camera_timestamp_base;
  guint64 last_frame_id;
  guint64 last_stream_frame_id;
  gboolean have_last_stream_frame_id;
  GstClockTime start_time;
  GstClockTime frame_duration;
  gboolean have_camera_timestamp_base;
  gboolean have_last_frame_id;
  gint stopping;
  gboolean acquisition_started;
  gboolean chunks_enabled;
  guint consecutive_timeouts;
  guint64 total_timeouts;
  guint64 total_incomplete;
  guint64 total_frame_gaps;
  guint64 total_crc_failures;
  guint64 actual_stream_buffer_count;
  gchar* resolved_serial;
  gchar* device_model;
  gchar* firmware_version;
  gchar* actual_pixel_format;
  gchar* enabled_chunks;
  guint64 timestamp_increment_ns;
  gboolean have_timestamp_latch;
  guint64 timestamp_latch_value;
  guint64 latch_host_monotonic_before_ns;
  guint64 latch_host_monotonic_after_ns;
  guint64 latch_host_unix_before_ns;
  guint64 latch_host_unix_after_ns;
  guint64 last_telemetry_frame;

  SystemPtr* system;
  CameraList* camera_list;
  CameraPtr* camera;
  ImageProcessor* processor;
};

struct _GstFlirSpinSrcClass {
  GstPushSrcClass parent_class;
};

}  // namespace

using GstFlirSpinSrc = _GstFlirSpinSrc;
using GstFlirSpinSrcClass = _GstFlirSpinSrcClass;

#define GST_TYPE_FLIR_SPIN_SRC (gst_flir_spin_src_get_type())
#define GST_FLIR_SPIN_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj), GST_TYPE_FLIR_SPIN_SRC, GstFlirSpinSrc))

G_DEFINE_TYPE(GstFlirSpinSrc, gst_flir_spin_src, GST_TYPE_PUSH_SRC)

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, "
        "format = (string) GRAY8, "
        "width = (int) [ 1, 2147483647 ], "
        "height = (int) [ 1, 2147483647 ], "
        "framerate = (fraction) [ 0/1, 2147483647/1 ]"));

static std::string spinnaker_exception_message(const Spinnaker::Exception& exc) {
  std::ostringstream oss;
  oss << exc.what();
  return oss.str();
}

static gboolean environment_value_is_truthy(const gchar* name) {
  const gchar* value = g_getenv(name);
  return value &&
      (g_ascii_strcasecmp(value, "1") == 0 ||
       g_ascii_strcasecmp(value, "true") == 0 ||
       g_ascii_strcasecmp(value, "yes") == 0 ||
       g_ascii_strcasecmp(value, "on") == 0);
}

static bool exception_is_timeout(const Spinnaker::Exception& exc) {
  return exc.GetError() == Spinnaker::SPINNAKER_ERR_TIMEOUT;
}

static std::string json_escape(const std::string& value) {
  std::ostringstream out;
  for (const unsigned char ch : value) {
    switch (ch) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\b': out << "\\b"; break;
      case '\f': out << "\\f"; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default:
        if (ch < 0x20) {
          out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << static_cast<int>(ch)
              << std::dec << std::setfill(' ');
        } else {
          out << ch;
        }
    }
  }
  return out.str();
}

static std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::ostringstream out;
  out << '"';
  for (const char ch : value) {
    if (ch == '"') {
      out << "\"\"";
    } else {
      out << ch;
    }
  }
  out << '"';
  return out.str();
}

static bool write_diagnostic_event(
    GstFlirSpinSrc* self,
    const char* event_type,
    guint64 host_unix_ns,
    guint64 host_monotonic_ns,
    const std::string& expected_frame_id,
    const std::string& actual_frame_id,
    const std::string& details) {
  if (!self->error_log) {
    return true;
  }
  std::ostringstream row;
  row << host_unix_ns << ',' << host_monotonic_ns << ','
      << event_type << ',' << self->camera_index << ','
      << expected_frame_id << ',' << actual_frame_id << ','
      << csv_escape(details) << '\n';
  const std::string line = row.str();
  return std::fwrite(line.data(), 1, line.size(), self->error_log) == line.size()
      && std::fflush(self->error_log) == 0;
}

static std::optional<std::string> string_get(INodeMap& node_map, const char* node_name) {
  CStringPtr node = node_map.GetNode(node_name);
  if (!IsAvailable(node) || !IsReadable(node)) {
    return std::nullopt;
  }
  return std::string(node->GetValue().c_str());
}

static std::optional<std::string> enum_get(INodeMap& node_map, const char* node_name) {
  CEnumerationPtr node = node_map.GetNode(node_name);
  if (!IsAvailable(node) || !IsReadable(node)) {
    return std::nullopt;
  }
  CEnumEntryPtr entry = node->GetCurrentEntry();
  if (!IsAvailable(entry) || !IsReadable(entry)) {
    return std::nullopt;
  }
  return std::string(entry->GetSymbolic().c_str());
}

static std::optional<double> float_get(INodeMap& node_map, const char* node_name) {
  CFloatPtr node = node_map.GetNode(node_name);
  if (!IsAvailable(node) || !IsReadable(node)) {
    return std::nullopt;
  }
  return node->GetValue();
}

static std::optional<guint64> integer_get(INodeMap& node_map, const char* node_name) {
  CIntegerPtr node = node_map.GetNode(node_name);
  if (!IsAvailable(node) || !IsReadable(node)) {
    return std::nullopt;
  }
  return static_cast<guint64>(node->GetValue());
}

static std::optional<double> selected_float_get(
    INodeMap& node_map,
    const char* selector_name,
    const char* entry_name,
    const char* value_name) {
  CEnumerationPtr selector = node_map.GetNode(selector_name);
  CFloatPtr value = node_map.GetNode(value_name);
  if (!IsAvailable(selector) || !IsReadable(selector) || !IsWritable(selector) ||
      !IsAvailable(value) || !IsReadable(value)) {
    return std::nullopt;
  }
  CEnumEntryPtr entry = selector->GetEntryByName(entry_name);
  if (!IsAvailable(entry) || !IsReadable(entry)) {
    return std::nullopt;
  }
  const int64_t previous = selector->GetIntValue();
  try {
    selector->SetIntValue(entry->GetValue());
    const double result = value->GetValue();
    selector->SetIntValue(previous);
    return result;
  } catch (...) {
    try { selector->SetIntValue(previous); } catch (...) {}
    return std::nullopt;
  }
}

static bool enum_set(INodeMap& node_map, const char* node_name, const char* entry_name, bool required) {
  CEnumerationPtr node = node_map.GetNode(node_name);
  if (IsAvailable(node) && IsReadable(node)) {
    CEnumEntryPtr current = node->GetCurrentEntry();
    if (IsAvailable(current) && IsReadable(current)) {
      const std::string current_name = current->GetSymbolic().c_str();
      if (current_name == entry_name) {
        return true;
      }
    }
  }

  if (!IsAvailable(node) || !IsWritable(node)) {
    if (required) {
      throw std::runtime_error(std::string("Node not available/writable and not already set: ") + node_name + "=" + entry_name);
    }
    return false;
  }

  CEnumEntryPtr entry = node->GetEntryByName(entry_name);
  if (!IsAvailable(entry) || !IsReadable(entry)) {
    if (required) {
      throw std::runtime_error(std::string("Entry not available/readable: ") + node_name + "=" + entry_name);
    }
    return false;
  }

  node->SetIntValue(entry->GetValue());
  return true;
}

static void latch_camera_clock(GstFlirSpinSrc* self, CameraPtr cam) {
  self->have_timestamp_latch = FALSE;
  INodeMap& node_map = cam->GetNodeMap();
  CCommandPtr latch = node_map.GetNode("TimestampLatch");
  CIntegerPtr value = node_map.GetNode("TimestampLatchValue");
  if (!IsAvailable(latch) || !IsWritable(latch) || !IsAvailable(value) || !IsReadable(value)) {
    return;
  }
  try {
    self->latch_host_monotonic_before_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    self->latch_host_unix_before_ns = static_cast<guint64>(g_get_real_time()) * 1000;
    latch->Execute();
    self->timestamp_latch_value = static_cast<guint64>(value->GetValue());
    self->latch_host_monotonic_after_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    self->latch_host_unix_after_ns = static_cast<guint64>(g_get_real_time()) * 1000;
    self->have_timestamp_latch = TRUE;
  } catch (const Spinnaker::Exception& exc) {
    GST_WARNING_OBJECT(self, "Unable to latch camera timestamp: %s", exc.what());
  }
}

static bool scientific_chunk(const std::string& name) {
  static const std::vector<std::string> wanted = {
      "CRC",
      "FrameID",
      "OffsetX",
      "OffsetY",
      "Width",
      "Height",
      "ExposureTime",
      "Gain",
      "BlackLevel",
      "PixelFormat",
      "Timestamp",
      "SequencerSetActive",
  };
  return std::find(wanted.begin(), wanted.end(), name) != wanted.end();
}

static void configure_chunk_data(GstFlirSpinSrc* self, CameraPtr cam) {
  INodeMap& node_map = cam->GetNodeMap();
  CBooleanPtr mode = node_map.GetNode("ChunkModeActive");
  const bool disabled = !self->metadata_profile ||
      g_ascii_strcasecmp(self->metadata_profile, "off") == 0;
  self->chunks_enabled = FALSE;
  g_clear_pointer(&self->enabled_chunks, g_free);
  self->enabled_chunks = g_strdup("");

  if (!IsAvailable(mode) || !IsWritable(mode)) {
    if (!disabled) {
      throw std::runtime_error("ChunkModeActive is unavailable or not writable");
    }
    return;
  }
  if (disabled) {
    mode->SetValue(false);
    return;
  }

  mode->SetValue(true);
  CEnumerationPtr selector = node_map.GetNode("ChunkSelector");
  CBooleanPtr enable = node_map.GetNode("ChunkEnable");
  if (!IsAvailable(selector) || !IsReadable(selector) || !IsAvailable(enable)) {
    throw std::runtime_error("ChunkSelector/ChunkEnable is unavailable");
  }

  Spinnaker::GenApi::NodeList_t entries;
  selector->GetEntries(entries);
  std::vector<std::string> enabled;
  for (const auto& raw_entry : entries) {
    CEnumEntryPtr entry = raw_entry;
    if (!IsAvailable(entry) || !IsReadable(entry)) {
      continue;
    }
    const std::string name = entry->GetSymbolic().c_str();
    if (!scientific_chunk(name)) {
      continue;
    }
    selector->SetIntValue(entry->GetValue());
    if (enable->GetValue() || IsWritable(enable)) {
      if (!enable->GetValue()) {
        enable->SetValue(true);
      }
      enabled.push_back(name);
    }
  }
  if (enabled.empty()) {
    throw std::runtime_error("No scientific chunk fields could be enabled");
  }
  std::ostringstream names;
  for (size_t index = 0; index < enabled.size(); ++index) {
    if (index) {
      names << ',';
    }
    names << enabled[index];
  }
  g_free(self->enabled_chunks);
  self->enabled_chunks = g_strdup(names.str().c_str());
  self->chunks_enabled = TRUE;
}

static gpointer flir_meta_copy(gpointer data, gpointer) {
  return data ? g_strdup(static_cast<const gchar*>(data)) : nullptr;
}

static void flir_meta_release(gpointer data, gpointer) {
  g_free(data);
}

static gpointer flir_gst_to_nvds_meta(gpointer data, gpointer) {
  NvDsUserMeta* user_meta = static_cast<NvDsUserMeta*>(data);
  return user_meta && user_meta->user_meta_data
      ? g_strdup(static_cast<const gchar*>(user_meta->user_meta_data))
      : nullptr;
}

static void flir_nvds_meta_release(gpointer data, gpointer) {
  NvDsUserMeta* user_meta = static_cast<NvDsUserMeta*>(data);
  if (user_meta) {
    g_clear_pointer(&user_meta->user_meta_data, g_free);
  }
}

static bool attach_flir_frame_meta(GstBuffer* buffer, const std::string& payload) {
  gchar* owned_payload = g_strdup(payload.c_str());
  NvDsMeta* meta = gst_buffer_add_nvds_meta(
      buffer,
      owned_payload,
      nullptr,
      flir_meta_copy,
      flir_meta_release);
  if (!meta) {
    g_free(owned_payload);
    return false;
  }
  meta->meta_type = static_cast<GstNvDsMetaType>(
      nvds_get_user_meta_type(const_cast<gchar*>(FLIR_FRAME_META_DESCRIPTOR)));
  meta->gst_to_nvds_meta_transform_func = flir_gst_to_nvds_meta;
  meta->gst_to_nvds_meta_release_func = flir_nvds_meta_release;
  return true;
}

static CameraPtr select_camera(GstFlirSpinSrc* self) {
  if (!self->camera_serial || !*self->camera_serial) {
    return self->camera_list->GetByIndex(self->camera_index);
  }
  for (unsigned int index = 0; index < self->camera_list->GetSize(); ++index) {
    CameraPtr candidate = self->camera_list->GetByIndex(index);
    auto serial = string_get(candidate->GetTLDeviceNodeMap(), "DeviceSerialNumber");
    if (serial && *serial == self->camera_serial) {
      self->camera_index = index;
      return candidate;
    }
  }
  throw std::runtime_error(std::string("No Spinnaker camera found with serial ") + self->camera_serial);
}

static void maybe_set_trigger(GstFlirSpinSrc* self, CameraPtr cam) {
  INodeMap& node_map = cam->GetNodeMap();

  try {
    cam->EndAcquisition();
  } catch (...) {
  }

  enum_set(node_map, "TriggerMode", "Off", true);

  if (!self->trigger) {
    return;
  }

  enum_set(node_map, "TriggerSelector", "FrameStart", true);

  CEnumerationPtr line_selector = node_map.GetNode("LineSelector");
  if (IsAvailable(line_selector) && IsWritable(line_selector)) {
    CEnumEntryPtr line0 = line_selector->GetEntryByName("Line0");
    if (IsAvailable(line0) && IsReadable(line0)) {
      line_selector->SetIntValue(line0->GetValue());
    }
  }

  enum_set(node_map, "LineMode", "Input", false);
  enum_set(node_map, "TriggerSource", "Line0", true);

  const char* activation = "RisingEdge";
  if (self->trigger_activation && g_ascii_strncasecmp(self->trigger_activation, "fall", 4) == 0) {
    activation = "FallingEdge";
  }
  enum_set(node_map, "TriggerActivation", activation, true);

  enum_set(node_map, "ExposureAuto", "Off", false);
  enum_set(node_map, "TriggerOverlap", "ReadOut", false);
  enum_set(node_map, "TriggerMode", "On", true);
}

static void configure_stream_buffers(GstFlirSpinSrc* self, CameraPtr cam) {
  self->actual_stream_buffer_count = 0;
  INodeMap& stream_map = cam->GetTLStreamNodeMap();
  if (self->buffer_handling && *self->buffer_handling) {
    enum_set(stream_map, "StreamBufferHandlingMode", self->buffer_handling, false);
  }
  if (self->stream_buffer_count == 0) {
    return;
  }
  enum_set(stream_map, "StreamBufferCountMode", "Manual", false);
  CIntegerPtr count = stream_map.GetNode("StreamBufferCountManual");
  if (!IsAvailable(count) || !IsWritable(count)) {
    GST_WARNING_OBJECT(self, "StreamBufferCountManual is unavailable; using camera default");
    return;
  }
  const int64_t minimum = count->GetMin();
  const int64_t maximum = count->GetMax();
  const int64_t desired = std::clamp<int64_t>(
      static_cast<int64_t>(self->stream_buffer_count), minimum, maximum);
  count->SetValue(desired);
  self->actual_stream_buffer_count = static_cast<guint64>(count->GetValue());
}

static void configure_camera(GstFlirSpinSrc* self, CameraPtr cam) {
  INodeMap& node_map = cam->GetNodeMap();

  configure_stream_buffers(self, cam);
  enum_set(node_map, "AcquisitionMode", "Continuous", false);

  if (self->pixel_format && *self->pixel_format) {
    enum_set(node_map, "PixelFormat", self->pixel_format, true);
  }

  CIntegerPtr width_node = node_map.GetNode("Width");
  if (IsAvailable(width_node) && IsWritable(width_node)) {
    const int64_t desired = static_cast<int64_t>(self->width);
    const int64_t minimum = width_node->GetMin();
    const int64_t maximum = width_node->GetMax();
    const int64_t increment = std::max<int64_t>(1, width_node->GetInc());
    if (desired < minimum || desired > maximum || (desired - minimum) % increment != 0) {
      std::ostringstream oss;
      oss << "Requested width " << desired << " is unsupported; range=" << minimum << ".."
          << maximum << " increment=" << increment;
      throw std::runtime_error(oss.str());
    }
    width_node->SetValue(desired);
  }

  CIntegerPtr height_node = node_map.GetNode("Height");
  if (IsAvailable(height_node) && IsWritable(height_node)) {
    const int64_t desired = static_cast<int64_t>(self->height);
    const int64_t minimum = height_node->GetMin();
    const int64_t maximum = height_node->GetMax();
    const int64_t increment = std::max<int64_t>(1, height_node->GetInc());
    if (desired < minimum || desired > maximum || (desired - minimum) % increment != 0) {
      std::ostringstream oss;
      oss << "Requested height " << desired << " is unsupported; range=" << minimum << ".."
          << maximum << " increment=" << increment;
      throw std::runtime_error(oss.str());
    }
    height_node->SetValue(desired);
  }

  CBooleanPtr fps_enable = node_map.GetNode("AcquisitionFrameRateEnable");
  CFloatPtr fps_node = node_map.GetNode("AcquisitionFrameRate");
  if (self->trigger) {
    if (IsAvailable(fps_enable) && IsWritable(fps_enable)) {
      fps_enable->SetValue(false);
    }
  } else {
    if (IsAvailable(fps_enable) && IsWritable(fps_enable)) {
      fps_enable->SetValue(true);
    }
    if (IsAvailable(fps_node) && IsWritable(fps_node)) {
      fps_node->SetValue(static_cast<double>(self->fps));
    }
  }

  if (self->exposure_us >= 0.0) {
    enum_set(node_map, "ExposureAuto", "Off", false);
    CFloatPtr exposure_node = node_map.GetNode("ExposureTime");
    if (IsAvailable(exposure_node) && IsWritable(exposure_node)) {
      if (self->exposure_us < exposure_node->GetMin() || self->exposure_us > exposure_node->GetMax()) {
        std::ostringstream oss;
        oss << "Requested exposure-us " << self->exposure_us << " is unsupported; range="
            << exposure_node->GetMin() << ".." << exposure_node->GetMax();
        throw std::runtime_error(oss.str());
      }
      exposure_node->SetValue(self->exposure_us);
    }
  }

  if (self->gain >= 0.0) {
    enum_set(node_map, "GainAuto", "Off", false);
    CFloatPtr gain_node = node_map.GetNode("Gain");
    if (IsAvailable(gain_node) && IsWritable(gain_node)) {
      if (self->gain < gain_node->GetMin() || self->gain > gain_node->GetMax()) {
        std::ostringstream oss;
        oss << "Requested gain " << self->gain << " is unsupported; range="
            << gain_node->GetMin() << ".." << gain_node->GetMax();
        throw std::runtime_error(oss.str());
      }
      gain_node->SetValue(self->gain);
    }
  }

  maybe_set_trigger(self, cam);
  configure_chunk_data(self, cam);

  self->actual_width = self->width;
  self->actual_height = self->height;
  self->actual_fps = self->fps;
  self->actual_exposure_us = -1.0;
  self->actual_gain_db = -1.0;
  if (IsAvailable(width_node) && IsReadable(width_node)) {
    self->actual_width = static_cast<guint>(width_node->GetValue());
  }
  if (IsAvailable(height_node) && IsReadable(height_node)) {
    self->actual_height = static_cast<guint>(height_node->GetValue());
  }
  if (IsAvailable(fps_node) && IsReadable(fps_node) && !self->trigger) {
    self->actual_fps = std::max(0.001, fps_node->GetValue());
  }
  if (const auto value = float_get(node_map, "ExposureTime")) {
    self->actual_exposure_us = *value;
  }
  if (const auto value = float_get(node_map, "Gain")) {
    self->actual_gain_db = *value;
  }
  if (const auto value = enum_get(node_map, "PixelFormat")) {
    g_free(self->actual_pixel_format);
    self->actual_pixel_format = g_strdup(value->c_str());
  }
  if (const auto value = integer_get(node_map, "TimestampIncrement")) {
    self->timestamp_increment_ns = *value;
  }
  if (const auto value = string_get(node_map, "DeviceModelName")) {
    g_free(self->device_model);
    self->device_model = g_strdup(value->c_str());
  }
  if (const auto value = string_get(node_map, "DeviceFirmwareVersion")) {
    g_free(self->firmware_version);
    self->firmware_version = g_strdup(value->c_str());
  }
}

static GstCaps* make_caps(GstFlirSpinSrc* self) {
  const guint width = self->actual_width ? self->actual_width : self->width;
  const guint height = self->actual_height ? self->actual_height : self->height;
  const guint fps = self->actual_fps > 0.0
      ? std::max<guint>(1, static_cast<guint>(std::llround(self->actual_fps)))
      : self->fps;
  return gst_caps_new_simple(
      "video/x-raw",
      "format",
      G_TYPE_STRING,
      "GRAY8",
      "width",
      G_TYPE_INT,
      static_cast<gint>(width),
      "height",
      G_TYPE_INT,
      static_cast<gint>(height),
      "framerate",
      GST_TYPE_FRACTION,
      static_cast<gint>(fps),
      1,
      nullptr);
}

static void close_camera(GstFlirSpinSrc* self) {
  g_atomic_int_set(&self->stopping, TRUE);

  try {
    if (self->camera && self->acquisition_started) {
      (*self->camera)->EndAcquisition();
    }
  } catch (...) {
  }
  self->acquisition_started = FALSE;

  try {
    if (self->camera) {
      (*self->camera)->DeInit();
    }
  } catch (...) {
  }

  delete self->camera;
  self->camera = nullptr;

  try {
    if (self->camera_list) {
      self->camera_list->Clear();
    }
  } catch (...) {
  }
  delete self->camera_list;
  self->camera_list = nullptr;

  try {
    if (self->system) {
      (*self->system)->ReleaseInstance();
    }
  } catch (...) {
  }
  delete self->system;
  self->system = nullptr;

  delete self->processor;
  self->processor = nullptr;

  if (self->capture_log) {
    std::fflush(self->capture_log);
    std::fclose(self->capture_log);
    self->capture_log = nullptr;
  }
  if (self->frame_manifest) {
    std::fflush(self->frame_manifest);
    std::fclose(self->frame_manifest);
    self->frame_manifest = nullptr;
  }
  if (self->camera_telemetry) {
    std::fflush(self->camera_telemetry);
    std::fclose(self->camera_telemetry);
    self->camera_telemetry = nullptr;
  }
  if (self->error_log) {
    std::fflush(self->error_log);
    std::fclose(self->error_log);
    self->error_log = nullptr;
  }
}

static gboolean open_camera(GstFlirSpinSrc* self, std::string& error) {
  try {
    if (self->capture_log_path && *self->capture_log_path) {
      self->capture_log = std::fopen(self->capture_log_path, "w");
      if (!self->capture_log) {
        error = std::string("Failed to open capture log: ") + self->capture_log_path;
        close_camera(self);
        return FALSE;
      }
      // Every emitted source buffer is a scientific audit record. Line buffering
      // preserves the ledger tail even when normal pipeline teardown is unavailable.
      std::setvbuf(self->capture_log, nullptr, _IOLBF, 0);
    }
    if (self->frame_manifest_path && *self->frame_manifest_path) {
      self->frame_manifest = std::fopen(self->frame_manifest_path, "w");
      if (!self->frame_manifest) {
        error = std::string("Failed to open frame manifest: ") + self->frame_manifest_path;
        close_camera(self);
        return FALSE;
      }
      // This is the canonical, source-owned frame index.  It is written while
      // acquisition is active so shutdown never has to reconstruct millions
      // of frame rows from the verbose JSON capture ledger.
      std::setvbuf(self->frame_manifest, nullptr, _IOFBF, 1024 * 1024);
      static constexpr const char* kFrameHeader =
          "stream_id,camera_serial,deepstream_frame_number,source_sequence_index,source,raw_frame_index,pts_ns,dts_ns,duration_ns,host_monotonic_ns,host_unix_ns,status,camera_frame_id,camera_frame_id_available,stream_frame_id,chunk_frame_id,frame_id_delta_consistent,missing_frames_before,pipeline_missing_frames_before,camera_timestamp_ns,chunk_timestamp_raw,timestamp_increment_ns,gst_pts_ns,timestamp_origin,host_received_monotonic_ns,host_received_unix_ns,copy_complete_monotonic_ns,observer_monotonic_ns,exposure_us,gain_db,black_level,payload_crc_valid,image_status,source_width,source_height,source_pixel_format,metadata_status,inference_admitted\n";
      if (std::fputs(kFrameHeader, self->frame_manifest) == EOF) {
        error = std::string("Failed to write frame manifest header: ") + self->frame_manifest_path;
        close_camera(self);
        return FALSE;
      }
    }
    if (self->camera_telemetry_path && *self->camera_telemetry_path) {
      self->camera_telemetry = std::fopen(self->camera_telemetry_path, "w");
      if (!self->camera_telemetry) {
        error = std::string("Failed to open camera telemetry: ") + self->camera_telemetry_path;
        close_camera(self);
        return FALSE;
      }
      std::setvbuf(self->camera_telemetry, nullptr, _IOLBF, 0);
      static constexpr const char* kTelemetryHeader =
          "host_unix_ns,host_monotonic_ns,stream_id,camera_serial,source_sequence_index,camera_frame_id,sensor_temperature_c,mainboard_temperature_c,stream_started_frames,stream_delivered_frames,stream_incomplete_frames,stream_lost_frames,stream_dropped_frames,stream_input_buffers,stream_output_buffers\n";
      if (std::fputs(kTelemetryHeader, self->camera_telemetry) == EOF) {
        error = std::string("Failed to write camera telemetry header: ") + self->camera_telemetry_path;
        close_camera(self);
        return FALSE;
      }
    }
    if (self->error_log_path && *self->error_log_path) {
      self->error_log = std::fopen(self->error_log_path, "w");
      if (!self->error_log) {
        error = std::string("Failed to open camera error log: ") + self->error_log_path;
        close_camera(self);
        return FALSE;
      }
      std::setvbuf(self->error_log, nullptr, _IOLBF, 0);
      static constexpr const char* kErrorHeader =
          "host_unix_ns,host_monotonic_ns,event_type,stream_id,expected_frame_id,actual_frame_id,details\n";
      if (std::fputs(kErrorHeader, self->error_log) == EOF) {
        error = std::string("Failed to write camera error header: ") + self->error_log_path;
        close_camera(self);
        return FALSE;
      }
    }
    self->system = new SystemPtr(System::GetInstance());
    self->camera_list = new CameraList((*self->system)->GetCameras());

    const unsigned int count = self->camera_list->GetSize();
    if (count == 0) {
      error = "No Spinnaker cameras found";
      close_camera(self);
      return FALSE;
    }
    if ((!self->camera_serial || !*self->camera_serial) && self->camera_index >= count) {
      std::ostringstream oss;
      oss << "camera-index " << self->camera_index << " out of range; found " << count << " camera(s)";
      error = oss.str();
      close_camera(self);
      return FALSE;
    }

    CameraPtr cam = select_camera(self);
    self->camera = new CameraPtr(cam);
    (*self->camera)->Init();

    auto serial = string_get((*self->camera)->GetTLDeviceNodeMap(), "DeviceSerialNumber");
    if (!serial) {
      serial = string_get((*self->camera)->GetNodeMap(), "DeviceSerialNumber");
    }
    g_free(self->resolved_serial);
    self->resolved_serial = g_strdup(serial ? serial->c_str() : "");

    configure_camera(self, *self->camera);
    latch_camera_clock(self, *self->camera);
    if (self->camera_runtime_path && *self->camera_runtime_path) {
      FILE* runtime = std::fopen(self->camera_runtime_path, "w");
      if (!runtime) {
        error = std::string("Failed to open camera runtime metadata: ") + self->camera_runtime_path;
        close_camera(self);
        return FALSE;
      }
      std::ostringstream document;
      document << std::setprecision(17)
          << "{\"schema_version\":\"1.0\",\"metadata_type\":\""
          << FLIR_FRAME_META_DESCRIPTOR << "\",\"cameras\":[{"
          << "\"camera_index\":" << self->camera_index
          << ",\"camera_serial\":\"" << json_escape(self->resolved_serial ? self->resolved_serial : "") << "\""
          << ",\"device_model\":\"" << json_escape(self->device_model ? self->device_model : "") << "\""
          << ",\"firmware_version\":\"" << json_escape(self->firmware_version ? self->firmware_version : "") << "\""
          << ",\"source_width\":" << self->actual_width
          << ",\"source_height\":" << self->actual_height
          << ",\"source_pixel_format\":\"" << json_escape(self->actual_pixel_format ? self->actual_pixel_format : "") << "\""
          << ",\"actual_fps\":" << self->actual_fps
          << ",\"configured_exposure_us\":" << self->actual_exposure_us
          << ",\"configured_gain_db\":" << self->actual_gain_db
          << ",\"configured_stream_buffer_count\":" << self->actual_stream_buffer_count
          << ",\"timestamp_increment_ns\":" << self->timestamp_increment_ns
          << ",\"timestamp_latch_available\":" << (self->have_timestamp_latch ? "true" : "false")
          << ",\"timestamp_latch_raw\":" << (self->have_timestamp_latch ? std::to_string(self->timestamp_latch_value) : "null")
          << ",\"timestamp_latch_host_monotonic_before_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_monotonic_before_ns) : "null")
          << ",\"timestamp_latch_host_monotonic_after_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_monotonic_after_ns) : "null")
          << ",\"timestamp_latch_host_unix_before_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_unix_before_ns) : "null")
          << ",\"timestamp_latch_host_unix_after_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_unix_after_ns) : "null")
          << ",\"enabled_chunks\":\"" << json_escape(self->enabled_chunks ? self->enabled_chunks : "") << "\"}]}\n";
      const std::string content = document.str();
      const bool wrote = std::fwrite(content.data(), 1, content.size(), runtime) == content.size();
      const bool closed = std::fclose(runtime) == 0;
      if (!wrote || !closed) {
        error = std::string("Failed to write camera runtime metadata: ") + self->camera_runtime_path;
        close_camera(self);
        return FALSE;
      }
    }
    self->processor = new ImageProcessor();
    self->processor->SetColorProcessing(Spinnaker::SPINNAKER_COLOR_PROCESSING_ALGORITHM_HQ_LINEAR);

    GstCaps* caps = make_caps(self);
    const gboolean caps_ok = gst_base_src_set_caps(GST_BASE_SRC(self), caps);
    gst_caps_unref(caps);
    if (!caps_ok) {
      error = "Failed to set source caps";
      close_camera(self);
      return FALSE;
    }

    (*self->camera)->BeginAcquisition();
    self->acquisition_started = TRUE;
    g_atomic_int_set(&self->stopping, FALSE);
    self->frame_count = 0;
    self->fault_injected = FALSE;
    self->camera_timestamp_base = 0;
    self->last_frame_id = 0;
    self->last_stream_frame_id = 0;
    self->have_last_stream_frame_id = FALSE;
    self->have_camera_timestamp_base = FALSE;
    self->have_last_frame_id = FALSE;
    self->consecutive_timeouts = 0;
    self->total_timeouts = 0;
    self->total_incomplete = 0;
    self->total_frame_gaps = 0;
    self->total_crc_failures = 0;
    self->last_telemetry_frame = G_MAXUINT64;
    self->start_time = gst_util_get_timestamp();
    self->frame_duration = !self->trigger && self->actual_fps > 0.0
        ? static_cast<GstClockTime>(static_cast<double>(GST_SECOND) / self->actual_fps)
        : GST_CLOCK_TIME_NONE;

    GST_INFO_OBJECT(
        self,
        "started FLIR camera index=%u serial=%s model=%s format=%s output=GRAY8 %ux%u@%.6f trigger=%s chunks=%s",
        self->camera_index,
        self->resolved_serial ? self->resolved_serial : "",
        self->device_model ? self->device_model : "",
        self->actual_pixel_format ? self->actual_pixel_format : "",
        self->actual_width,
        self->actual_height,
        self->actual_fps,
        self->trigger ? "true" : "false",
        self->enabled_chunks ? self->enabled_chunks : "");
    return TRUE;
  } catch (const Spinnaker::Exception& exc) {
    error = spinnaker_exception_message(exc);
  } catch (const std::exception& exc) {
    error = exc.what();
  } catch (...) {
    error = "Unknown error while opening Spinnaker camera";
  }

  close_camera(self);
  return FALSE;
}

static void gst_flir_spin_src_set_property(GObject* object, guint prop_id, const GValue* value, GParamSpec* pspec) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(object);

  switch (prop_id) {
    case PROP_CAMERA_INDEX:
      self->camera_index = g_value_get_uint(value);
      break;
    case PROP_WIDTH:
      self->width = g_value_get_uint(value);
      break;
    case PROP_HEIGHT:
      self->height = g_value_get_uint(value);
      break;
    case PROP_FPS:
      self->fps = g_value_get_uint(value);
      break;
    case PROP_PIXEL_FORMAT:
      g_free(self->pixel_format);
      self->pixel_format = g_value_dup_string(value);
      break;
    case PROP_TRIGGER:
      self->trigger = g_value_get_boolean(value);
      break;
    case PROP_TRIGGER_ACTIVATION:
      g_free(self->trigger_activation);
      self->trigger_activation = g_value_dup_string(value);
      break;
    case PROP_EXPOSURE_US:
      self->exposure_us = g_value_get_double(value);
      break;
    case PROP_GAIN:
      self->gain = g_value_get_double(value);
      break;
    case PROP_TIMEOUT_MS:
      self->timeout_ms = g_value_get_uint(value);
      break;
    case PROP_DROP_INCOMPLETE:
      self->drop_incomplete = g_value_get_boolean(value);
      break;
    case PROP_BUFFER_HANDLING:
      g_free(self->buffer_handling);
      self->buffer_handling = g_value_dup_string(value);
      break;
    case PROP_STREAM_BUFFER_COUNT:
      self->stream_buffer_count = g_value_get_uint(value);
      break;
    case PROP_CAMERA_SERIAL:
      g_free(self->camera_serial);
      self->camera_serial = g_value_dup_string(value);
      break;
    case PROP_METADATA_PROFILE:
      g_free(self->metadata_profile);
      self->metadata_profile = g_value_dup_string(value);
      break;
    case PROP_MAX_CONSECUTIVE_TIMEOUTS:
      self->max_consecutive_timeouts = g_value_get_uint(value);
      break;
    case PROP_CAPTURE_LOG_PATH:
      g_free(self->capture_log_path);
      self->capture_log_path = g_value_dup_string(value);
      break;
    case PROP_FRAME_MANIFEST_PATH:
      g_free(self->frame_manifest_path);
      self->frame_manifest_path = g_value_dup_string(value);
      break;
    case PROP_CAMERA_TELEMETRY_PATH:
      g_free(self->camera_telemetry_path);
      self->camera_telemetry_path = g_value_dup_string(value);
      break;
    case PROP_ERROR_LOG_PATH:
      g_free(self->error_log_path);
      self->error_log_path = g_value_dup_string(value);
      break;
    case PROP_CAMERA_RUNTIME_PATH:
      g_free(self->camera_runtime_path);
      self->camera_runtime_path = g_value_dup_string(value);
      break;
    case PROP_FAULT_AFTER_FRAMES:
      self->fault_after_frames = g_value_get_uint64(value);
      break;
    case PROP_FAULT_KIND:
      g_free(self->fault_kind);
      self->fault_kind = g_value_dup_string(value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void gst_flir_spin_src_get_property(GObject* object, guint prop_id, GValue* value, GParamSpec* pspec) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(object);

  switch (prop_id) {
    case PROP_CAMERA_INDEX:
      g_value_set_uint(value, self->camera_index);
      break;
    case PROP_WIDTH:
      g_value_set_uint(value, self->width);
      break;
    case PROP_HEIGHT:
      g_value_set_uint(value, self->height);
      break;
    case PROP_FPS:
      g_value_set_uint(value, self->fps);
      break;
    case PROP_PIXEL_FORMAT:
      g_value_set_string(value, self->pixel_format);
      break;
    case PROP_TRIGGER:
      g_value_set_boolean(value, self->trigger);
      break;
    case PROP_TRIGGER_ACTIVATION:
      g_value_set_string(value, self->trigger_activation);
      break;
    case PROP_EXPOSURE_US:
      g_value_set_double(value, self->exposure_us);
      break;
    case PROP_GAIN:
      g_value_set_double(value, self->gain);
      break;
    case PROP_TIMEOUT_MS:
      g_value_set_uint(value, self->timeout_ms);
      break;
    case PROP_DROP_INCOMPLETE:
      g_value_set_boolean(value, self->drop_incomplete);
      break;
    case PROP_BUFFER_HANDLING:
      g_value_set_string(value, self->buffer_handling);
      break;
    case PROP_STREAM_BUFFER_COUNT:
      g_value_set_uint(value, self->stream_buffer_count);
      break;
    case PROP_CAMERA_SERIAL:
      g_value_set_string(value, self->camera_serial);
      break;
    case PROP_METADATA_PROFILE:
      g_value_set_string(value, self->metadata_profile);
      break;
    case PROP_MAX_CONSECUTIVE_TIMEOUTS:
      g_value_set_uint(value, self->max_consecutive_timeouts);
      break;
    case PROP_CAPTURE_LOG_PATH:
      g_value_set_string(value, self->capture_log_path);
      break;
    case PROP_FRAME_MANIFEST_PATH:
      g_value_set_string(value, self->frame_manifest_path);
      break;
    case PROP_CAMERA_TELEMETRY_PATH:
      g_value_set_string(value, self->camera_telemetry_path);
      break;
    case PROP_ERROR_LOG_PATH:
      g_value_set_string(value, self->error_log_path);
      break;
    case PROP_CAMERA_RUNTIME_PATH:
      g_value_set_string(value, self->camera_runtime_path);
      break;
    case PROP_FAULT_AFTER_FRAMES:
      g_value_set_uint64(value, self->fault_after_frames);
      break;
    case PROP_FAULT_KIND:
      g_value_set_string(value, self->fault_kind);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void gst_flir_spin_src_finalize(GObject* object) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(object);
  close_camera(self);
  g_free(self->pixel_format);
  g_free(self->trigger_activation);
  g_free(self->buffer_handling);
  g_free(self->camera_serial);
  g_free(self->metadata_profile);
  g_free(self->capture_log_path);
  g_free(self->frame_manifest_path);
  g_free(self->camera_telemetry_path);
  g_free(self->error_log_path);
  g_free(self->camera_runtime_path);
  g_free(self->fault_kind);
  g_free(self->resolved_serial);
  g_free(self->device_model);
  g_free(self->firmware_version);
  g_free(self->actual_pixel_format);
  g_free(self->enabled_chunks);
  G_OBJECT_CLASS(gst_flir_spin_src_parent_class)->finalize(object);
}

static GstCaps* gst_flir_spin_src_get_caps(GstBaseSrc* base_src, GstCaps* filter) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(base_src);
  GstCaps* caps = make_caps(self);
  if (!filter) {
    return caps;
  }

  GstCaps* intersected = gst_caps_intersect_full(filter, caps, GST_CAPS_INTERSECT_FIRST);
  gst_caps_unref(caps);
  return intersected;
}

static gboolean gst_flir_spin_src_start(GstBaseSrc* base_src) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(base_src);
  self->latency_reference_enabled =
      environment_value_is_truthy("NVDS_ENABLE_LATENCY_MEASUREMENT");
  std::string error;
  if (!open_camera(self, error)) {
    GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ, ("Failed to open FLIR Spinnaker camera"), ("%s", error.c_str()));
    return FALSE;
  }
  return TRUE;
}

static gboolean gst_flir_spin_src_stop(GstBaseSrc* base_src) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(base_src);
  close_camera(self);
  return TRUE;
}

static gboolean gst_flir_spin_src_unlock(GstBaseSrc* base_src) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(base_src);
  g_atomic_int_set(&self->stopping, TRUE);
  return TRUE;
}

static gboolean gst_flir_spin_src_unlock_stop(GstBaseSrc* base_src) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(base_src);
  if (self->acquisition_started) {
    g_atomic_int_set(&self->stopping, FALSE);
  }
  return TRUE;
}

static GstFlowReturn copy_image_to_buffer(
    GstFlirSpinSrc* self,
    const ImagePtr& image,
    guint64 host_received_monotonic_ns,
    guint64 host_received_unix_ns,
    GstBuffer** out_buffer) {
  ImagePtr mono;
  if (image->GetPixelFormat() != PixelFormat_Mono8) {
    mono = self->processor->Convert(image, PixelFormat_Mono8);
  } else {
    mono = image;
  }

  const size_t output_width = mono->GetWidth();
  const size_t output_height = mono->GetHeight();
  const size_t output_stride = mono->GetStride();
  const size_t expected_width = self->actual_width;
  const size_t expected_height = self->actual_height;
  if (output_width != expected_width || output_height != expected_height) {
    GST_ELEMENT_ERROR(
        self,
        STREAM,
        FORMAT,
        ("Camera frame dimensions changed"),
        ("Got %zux%zu, expected %zux%zu", output_width, output_height, expected_width, expected_height));
    return GST_FLOW_ERROR;
  }

  const size_t output_size = expected_width * expected_height;
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, output_size, nullptr);
  if (!buffer) {
    return GST_FLOW_ERROR;
  }
  GstMapInfo map;
  if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }
  const auto* src = static_cast<const guint8*>(mono->GetData());
  auto* dst = static_cast<guint8*>(map.data);
  if (output_stride == expected_width) {
    std::memcpy(dst, src, output_size);
  } else {
    for (size_t row = 0; row < expected_height; ++row) {
      std::memcpy(dst + row * expected_width, src + row * output_stride, expected_width);
    }
  }
  gst_buffer_unmap(buffer, &map);
  const guint64 copy_complete_monotonic_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;

  std::optional<guint64> transport_frame_id;
  std::optional<guint64> transport_timestamp_ns;
  try {
    transport_frame_id = static_cast<guint64>(image->GetFrameID());
  } catch (...) {
  }
  try {
    const guint64 value = static_cast<guint64>(image->GetTimeStamp());
    if (value > 0) {
      transport_timestamp_ns = value;
    }
  } catch (...) {
  }

  std::optional<guint64> chunk_frame_id;
  std::optional<gint64> chunk_timestamp_raw;
  std::optional<double> chunk_exposure_us;
  std::optional<double> chunk_gain_db;
  std::optional<double> chunk_black_level;
  std::optional<gint64> chunk_width;
  std::optional<gint64> chunk_height;
  std::optional<gint64> chunk_offset_x;
  std::optional<gint64> chunk_offset_y;
  std::optional<gint64> chunk_sequencer_set;
  if (self->chunks_enabled) {
    try {
      const ChunkData& chunk = image->GetChunkData();
      try { const auto value = chunk.GetFrameID(); if (value >= 0) chunk_frame_id = static_cast<guint64>(value); } catch (...) {}
      try { chunk_timestamp_raw = chunk.GetTimestamp(); } catch (...) {}
      try { chunk_exposure_us = chunk.GetExposureTime(); } catch (...) {}
      try { chunk_gain_db = chunk.GetGain(); } catch (...) {}
      try { chunk_black_level = chunk.GetBlackLevel(); } catch (...) {}
      try { chunk_width = chunk.GetWidth(); } catch (...) {}
      try { chunk_height = chunk.GetHeight(); } catch (...) {}
      try { chunk_offset_x = chunk.GetOffsetX(); } catch (...) {}
      try { chunk_offset_y = chunk.GetOffsetY(); } catch (...) {}
      try { chunk_sequencer_set = chunk.GetSequencerSetActive(); } catch (...) {}
    } catch (...) {
    }
  }

  const std::optional<guint64> camera_frame_id = chunk_frame_id;
  std::optional<bool> frame_id_delta_consistent;
  if (transport_frame_id && chunk_frame_id && self->have_last_stream_frame_id && self->have_last_frame_id) {
    if (*transport_frame_id >= self->last_stream_frame_id && *chunk_frame_id >= self->last_frame_id) {
      frame_id_delta_consistent =
          (*transport_frame_id - self->last_stream_frame_id) == (*chunk_frame_id - self->last_frame_id);
    } else {
      frame_id_delta_consistent = false;
    }
  }
  guint64 missing_frames_before = 0;
  gboolean discontinuity = FALSE;
  if (camera_frame_id && self->have_last_frame_id && *camera_frame_id != self->last_frame_id + 1) {
    const guint64 expected = self->last_frame_id + 1;
    if (*camera_frame_id > expected) {
      missing_frames_before = *camera_frame_id - expected;
    }
    discontinuity = TRUE;
    ++self->total_frame_gaps;
    GST_WARNING_OBJECT(
        self,
        "FLIR frame id gap on camera %u: expected %" G_GUINT64_FORMAT ", got %" G_GUINT64_FORMAT,
        self->camera_index,
        expected,
        *camera_frame_id);
    gst_element_post_message(
        GST_ELEMENT(self),
        gst_message_new_element(
            GST_OBJECT(self),
            gst_structure_new(
                "flir-frame-gap",
                "camera-index",
                G_TYPE_UINT,
                self->camera_index,
                "expected-frame-id",
                G_TYPE_UINT64,
                expected,
                "actual-frame-id",
                G_TYPE_UINT64,
                *camera_frame_id,
                nullptr)));
    if (!write_diagnostic_event(
            self,
            "camera_frame_gap",
            host_received_unix_ns,
            host_received_monotonic_ns,
            std::to_string(expected),
            std::to_string(*camera_frame_id),
            std::string("{\"missing_frames\":") +
                std::to_string(missing_frames_before) + "}")) {
      GST_ELEMENT_ERROR(
          self, RESOURCE, WRITE, ("Failed to write camera diagnostic event"),
          ("path=%s", self->error_log_path ? self->error_log_path : ""));
      gst_buffer_unref(buffer);
      return GST_FLOW_ERROR;
    }
  }
  if (camera_frame_id) {
    self->last_frame_id = *camera_frame_id;
    self->have_last_frame_id = TRUE;
  }

  GstClockTime pts = GST_CLOCK_TIME_NONE;
  if (transport_frame_id) {
    self->last_stream_frame_id = *transport_frame_id;
    self->have_last_stream_frame_id = TRUE;
  }
  const char* timestamp_origin = "host_monotonic_fallback";
  if (transport_timestamp_ns) {
    if (!self->have_camera_timestamp_base) {
      self->camera_timestamp_base = *transport_timestamp_ns;
      self->have_camera_timestamp_base = TRUE;
    }
    if (*transport_timestamp_ns >= self->camera_timestamp_base) {
      pts = *transport_timestamp_ns - self->camera_timestamp_base;
      timestamp_origin = "flir_transport";
    }
  }
  if (!GST_CLOCK_TIME_IS_VALID(pts)) {
    pts = host_received_monotonic_ns >= self->start_time
        ? host_received_monotonic_ns - self->start_time
        : 0;
  }

  GST_BUFFER_PTS(buffer) = pts;
  GST_BUFFER_DTS(buffer) = pts;
  GST_BUFFER_DURATION(buffer) = self->frame_duration;
  if (self->latency_reference_enabled) {
    const gchar* source_element_name = GST_ELEMENT_NAME(self);
    if (!source_element_name || !*source_element_name || self->frame_count > G_MAXUINT) {
      GST_ELEMENT_ERROR(
          self,
          STREAM,
          FAILED,
          ("Cannot add the required DeepStream latency reference timestamp"),
          ("source=%s source-sequence=%" G_GUINT64_FORMAT,
           source_element_name ? source_element_name : "",
           self->frame_count));
      gst_buffer_unref(buffer);
      return GST_FLOW_ERROR;
    }
    // DeepStream 9.1 exposes no return value for this public API. The call is
    // therefore mandatory in the debug path; downstream qualification remains
    // fail-closed unless NVIDIA's probes emit valid latency records.
    nvds_add_reference_timestamp_meta(
        buffer,
        const_cast<gchar*>(source_element_name),
        static_cast<guint>(self->frame_count));
  }
  if (camera_frame_id) {
    GST_BUFFER_OFFSET(buffer) = *camera_frame_id;
    GST_BUFFER_OFFSET_END(buffer) = *camera_frame_id + 1;
  } else {
    GST_BUFFER_OFFSET(buffer) = GST_BUFFER_OFFSET_NONE;
    GST_BUFFER_OFFSET_END(buffer) = GST_BUFFER_OFFSET_NONE;
  }
  if (discontinuity) {
    GST_BUFFER_FLAG_SET(buffer, GST_BUFFER_FLAG_DISCONT);
  }

  bool crc_checked = false;
  bool crc_valid = false;
  if (self->chunks_enabled && self->enabled_chunks && std::strstr(self->enabled_chunks, "CRC")) {
    try {
      crc_valid = image->CheckCRC();
      crc_checked = true;
      if (!crc_valid) {
        ++self->total_crc_failures;
        if (!write_diagnostic_event(
                self,
                "payload_crc_failure",
                host_received_unix_ns,
                host_received_monotonic_ns,
                "",
                camera_frame_id ? std::to_string(*camera_frame_id) : "",
                "{}")) {
          GST_ELEMENT_ERROR(
              self, RESOURCE, WRITE, ("Failed to write camera diagnostic event"),
              ("path=%s", self->error_log_path ? self->error_log_path : ""));
          gst_buffer_unref(buffer);
          return GST_FLOW_ERROR;
        }
      }
    } catch (...) {
    }
  }

  const guint64 telemetry_interval = std::max<guint64>(
      1, static_cast<guint64>(std::llround(std::max(1.0, self->actual_fps))));
  const bool telemetry_sample = self->last_telemetry_frame == G_MAXUINT64 ||
      self->frame_count - self->last_telemetry_frame >= telemetry_interval;
  std::optional<double> sensor_temperature_c;
  std::optional<double> mainboard_temperature_c;
  std::optional<guint64> stream_started_frames;
  std::optional<guint64> stream_delivered_frames;
  std::optional<guint64> stream_incomplete_frames;
  std::optional<guint64> stream_lost_frames;
  std::optional<guint64> stream_dropped_frames;
  std::optional<guint64> stream_input_buffers;
  std::optional<guint64> stream_output_buffers;
  if (telemetry_sample && self->camera) {
    INodeMap& node_map = (*self->camera)->GetNodeMap();
    INodeMap& stream_map = (*self->camera)->GetTLStreamNodeMap();
    sensor_temperature_c = selected_float_get(
        node_map, "DeviceTemperatureSelector", "Sensor", "DeviceTemperature");
    mainboard_temperature_c = selected_float_get(
        node_map, "DeviceTemperatureSelector", "Mainboard", "DeviceTemperature");
    stream_started_frames = integer_get(stream_map, "StreamStartedFrameCount");
    stream_delivered_frames = integer_get(stream_map, "StreamDeliveredFrameCount");
    stream_incomplete_frames = integer_get(stream_map, "StreamIncompleteFrameCount");
    stream_lost_frames = integer_get(stream_map, "StreamLostFrameCount");
    stream_dropped_frames = integer_get(stream_map, "StreamDroppedFrameCount");
    stream_input_buffers = integer_get(stream_map, "StreamInputBufferCount");
    stream_output_buffers = integer_get(stream_map, "StreamOutputBufferCount");
    self->last_telemetry_frame = self->frame_count;
  }
  if (telemetry_sample && self->camera_telemetry) {
    const auto uint_field = [](const std::optional<guint64>& value) {
      return value ? std::to_string(*value) : std::string();
    };
    const auto double_field = [](const std::optional<double>& value) {
      if (!value) {
        return std::string();
      }
      std::ostringstream text;
      text << std::setprecision(17) << *value;
      return text.str();
    };
    std::ostringstream row;
    row << host_received_unix_ns << ',' << host_received_monotonic_ns << ','
        << self->camera_index << ','
        << csv_escape(self->resolved_serial ? self->resolved_serial : "") << ','
        << self->frame_count << ',' << uint_field(camera_frame_id) << ','
        << double_field(sensor_temperature_c) << ','
        << double_field(mainboard_temperature_c) << ','
        << uint_field(stream_started_frames) << ','
        << uint_field(stream_delivered_frames) << ','
        << uint_field(stream_incomplete_frames) << ','
        << uint_field(stream_lost_frames) << ','
        << uint_field(stream_dropped_frames) << ','
        << uint_field(stream_input_buffers) << ','
        << uint_field(stream_output_buffers) << '\n';
    const std::string line = row.str();
    if (std::fwrite(line.data(), 1, line.size(), self->camera_telemetry) != line.size()) {
      GST_ELEMENT_ERROR(
          self, RESOURCE, WRITE, ("Failed to write camera telemetry"),
          ("path=%s", self->camera_telemetry_path ? self->camera_telemetry_path : ""));
      gst_buffer_unref(buffer);
      return GST_FLOW_ERROR;
    }
  }

  const auto image_status = image->GetImageStatus();
  const char* image_status_description = Spinnaker::Image::GetImageStatusDescription(image_status);
  std::ostringstream metadata;
  metadata << std::setprecision(17)
      << "{\"schema_version\":1"
      << ",\"meta_type\":\"" << FLIR_FRAME_META_DESCRIPTOR << "\""
      << ",\"camera_index\":" << self->camera_index
      << ",\"camera_serial\":\"" << json_escape(self->resolved_serial ? self->resolved_serial : "") << "\""
      << ",\"device_model\":\"" << json_escape(self->device_model ? self->device_model : "") << "\""
      << ",\"firmware_version\":\"" << json_escape(self->firmware_version ? self->firmware_version : "") << "\""
      << ",\"source_sequence_index\":" << self->frame_count
      << ",\"spinnaker_image_id\":" << image->GetID()
      << ",\"stream_index\":" << image->GetStreamIndex()
      << ",\"stream_frame_id\":" << (transport_frame_id ? std::to_string(*transport_frame_id) : "null")
      << ",\"chunk_frame_id\":" << (chunk_frame_id ? std::to_string(*chunk_frame_id) : "null")
      << ",\"camera_frame_id\":" << (camera_frame_id ? std::to_string(*camera_frame_id) : "null")
      << ",\"frame_id_delta_consistent\":"
      << (frame_id_delta_consistent ? (*frame_id_delta_consistent ? "true" : "false") : "null")
      << ",\"missing_frames_before\":" << missing_frames_before
      << ",\"transport_timestamp_ns\":" << (transport_timestamp_ns ? std::to_string(*transport_timestamp_ns) : "null")
      << ",\"chunk_timestamp_raw\":" << (chunk_timestamp_raw ? std::to_string(*chunk_timestamp_raw) : "null")
      << ",\"timestamp_increment_ns\":" << self->timestamp_increment_ns
      << ",\"timestamp_latch_available\":" << (self->have_timestamp_latch ? "true" : "false")
      << ",\"timestamp_latch_raw\":" << (self->have_timestamp_latch ? std::to_string(self->timestamp_latch_value) : "null")
      << ",\"timestamp_latch_host_monotonic_before_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_monotonic_before_ns) : "null")
      << ",\"timestamp_latch_host_monotonic_after_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_monotonic_after_ns) : "null")
      << ",\"timestamp_latch_host_unix_before_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_unix_before_ns) : "null")
      << ",\"timestamp_latch_host_unix_after_ns\":" << (self->have_timestamp_latch ? std::to_string(self->latch_host_unix_after_ns) : "null")
      << ",\"gst_pts_ns\":" << pts
      << ",\"timestamp_origin\":\"" << timestamp_origin << "\""
      << ",\"host_received_monotonic_ns\":" << host_received_monotonic_ns
      << ",\"host_received_unix_ns\":" << host_received_unix_ns
      << ",\"copy_complete_monotonic_ns\":" << copy_complete_monotonic_ns
      << ",\"source_width\":" << image->GetWidth()
      << ",\"source_height\":" << image->GetHeight()
      << ",\"source_offset_x\":" << image->GetXOffset()
      << ",\"source_offset_y\":" << image->GetYOffset()
      << ",\"source_stride_bytes\":" << image->GetStride()
      << ",\"source_pixel_format\":\"" << json_escape(image->GetPixelFormatName().c_str()) << "\""
      << ",\"bits_per_pixel\":" << image->GetBitsPerPixel()
      << ",\"channels\":" << image->GetNumChannels()
      << ",\"image_size_bytes\":" << image->GetImageSize()
      << ",\"valid_payload_size_bytes\":" << image->GetValidPayloadSize()
      << ",\"buffer_size_bytes\":" << image->GetBufferSize()
      << ",\"output_pixel_format\":\"GRAY8\""
      << ",\"output_size_bytes\":" << output_size
      << ",\"image_status_code\":" << static_cast<int>(image_status)
      << ",\"image_status\":\"" << json_escape(image_status_description ? image_status_description : "") << "\""
      << ",\"crc_checked\":" << (crc_checked ? "true" : "false")
      << ",\"crc_valid\":" << (crc_checked ? (crc_valid ? "true" : "false") : "null")
      << ",\"chunk_exposure_us\":" << (chunk_exposure_us ? std::to_string(*chunk_exposure_us) : "null")
      << ",\"chunk_gain_db\":" << (chunk_gain_db ? std::to_string(*chunk_gain_db) : "null")
      << ",\"chunk_black_level\":" << (chunk_black_level ? std::to_string(*chunk_black_level) : "null")
      << ",\"chunk_width\":" << (chunk_width ? std::to_string(*chunk_width) : "null")
      << ",\"chunk_height\":" << (chunk_height ? std::to_string(*chunk_height) : "null")
      << ",\"chunk_offset_x\":" << (chunk_offset_x ? std::to_string(*chunk_offset_x) : "null")
      << ",\"chunk_offset_y\":" << (chunk_offset_y ? std::to_string(*chunk_offset_y) : "null")
      << ",\"chunk_sequencer_set\":" << (chunk_sequencer_set ? std::to_string(*chunk_sequencer_set) : "null")
      << ",\"configured_exposure_us\":" << self->actual_exposure_us
      << ",\"configured_gain_db\":" << self->actual_gain_db
      << ",\"actual_fps\":" << self->actual_fps
      << ",\"configured_stream_buffer_count\":" << self->actual_stream_buffer_count
      << ",\"enabled_chunks\":\"" << json_escape(self->enabled_chunks ? self->enabled_chunks : "") << "\""
      << ",\"total_timeouts\":" << self->total_timeouts
      << ",\"total_incomplete\":" << self->total_incomplete
      << ",\"total_frame_gap_events\":" << self->total_frame_gaps
      << ",\"total_crc_failures\":" << self->total_crc_failures
      << ",\"telemetry_sample\":" << (telemetry_sample ? "true" : "false")
      << ",\"sensor_temperature_c\":" << (sensor_temperature_c ? std::to_string(*sensor_temperature_c) : "null")
      << ",\"mainboard_temperature_c\":" << (mainboard_temperature_c ? std::to_string(*mainboard_temperature_c) : "null")
      << ",\"stream_started_frames\":" << (stream_started_frames ? std::to_string(*stream_started_frames) : "null")
      << ",\"stream_delivered_frames\":" << (stream_delivered_frames ? std::to_string(*stream_delivered_frames) : "null")
      << ",\"stream_incomplete_frames\":" << (stream_incomplete_frames ? std::to_string(*stream_incomplete_frames) : "null")
      << ",\"stream_lost_frames\":" << (stream_lost_frames ? std::to_string(*stream_lost_frames) : "null")
      << ",\"stream_dropped_frames\":" << (stream_dropped_frames ? std::to_string(*stream_dropped_frames) : "null")
      << ",\"stream_input_buffers\":" << (stream_input_buffers ? std::to_string(*stream_input_buffers) : "null")
      << ",\"stream_output_buffers\":" << (stream_output_buffers ? std::to_string(*stream_output_buffers) : "null")
      << "}";
  if (!attach_flir_frame_meta(buffer, metadata.str())) {
    GST_ELEMENT_ERROR(self, STREAM, FAILED, ("Failed to attach FLIR frame metadata"), (nullptr));
    gst_buffer_unref(buffer);
    return GST_FLOW_ERROR;
  }
  if (self->capture_log) {
    const std::string line = metadata.str();
    if (std::fwrite(line.data(), 1, line.size(), self->capture_log) != line.size() ||
        std::fputc('\n', self->capture_log) == EOF) {
      GST_ELEMENT_ERROR(
          self,
          RESOURCE,
          WRITE,
          ("Failed to write FLIR capture ledger"),
          ("path=%s", self->capture_log_path ? self->capture_log_path : ""));
      gst_buffer_unref(buffer);
      return GST_FLOW_ERROR;
    }
  }
  if (self->frame_manifest) {
    const auto uint_field = [](const std::optional<guint64>& value) {
      return value ? std::to_string(*value) : std::string();
    };
    const auto double_field = [](const std::optional<double>& value) {
      if (!value) {
        return std::string();
      }
      std::ostringstream text;
      text << std::setprecision(17) << *value;
      return text.str();
    };
    std::ostringstream row;
    row << self->camera_index << ','
        << csv_escape(self->resolved_serial ? self->resolved_serial : "") << ','
        << self->frame_count << ','
        << self->frame_count << ','
        << "flirspinsrc:" << self->camera_index << ','
        << self->frame_count << ','
        << pts << ','
        << pts << ','
        << self->frame_duration << ','
        << host_received_monotonic_ns << ','
        << host_received_unix_ns << ','
        << "ok,"
        << uint_field(camera_frame_id) << ','
        << (camera_frame_id ? "1" : "0") << ','
        << uint_field(transport_frame_id) << ','
        << uint_field(chunk_frame_id) << ','
        << (frame_id_delta_consistent
                ? (*frame_id_delta_consistent ? "1" : "0")
                : "") << ','
        << missing_frames_before << ','
        << 0 << ','
        << uint_field(transport_timestamp_ns) << ','
        << uint_field(chunk_timestamp_raw) << ','
        << self->timestamp_increment_ns << ','
        << pts << ','
        << timestamp_origin << ','
        << host_received_monotonic_ns << ','
        << host_received_unix_ns << ','
        << copy_complete_monotonic_ns << ','
        << copy_complete_monotonic_ns << ','
        << double_field(chunk_exposure_us) << ','
        << double_field(chunk_gain_db) << ','
        << double_field(chunk_black_level) << ','
        << (crc_checked ? (crc_valid ? "1" : "0") : "") << ','
        << csv_escape(image_status_description ? image_status_description : "") << ','
        << image->GetWidth() << ','
        << image->GetHeight() << ','
        << csv_escape(image->GetPixelFormatName().c_str()) << ','
        << "ok,"  // metadata_status, inference_admitted
        << '\n';
    const std::string line = row.str();
    const bool flush_due = (self->frame_count + 1) % 100 == 0;
    if (std::fwrite(line.data(), 1, line.size(), self->frame_manifest) != line.size() ||
        (flush_due && std::fflush(self->frame_manifest) != 0)) {
      GST_ELEMENT_ERROR(
          self,
          RESOURCE,
          WRITE,
          ("Failed to write canonical frame manifest"),
          ("path=%s", self->frame_manifest_path ? self->frame_manifest_path : ""));
      gst_buffer_unref(buffer);
      return GST_FLOW_ERROR;
    }
  }
  ++self->frame_count;

  *out_buffer = buffer;
  return GST_FLOW_OK;
}

static GstFlowReturn gst_flir_spin_src_create(GstPushSrc* push_src, GstBuffer** out_buffer) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(push_src);
  if (!self->camera || !self->acquisition_started) {
    return GST_FLOW_FLUSHING;
  }

  if (self->fault_after_frames > 0 && !self->fault_injected &&
      self->frame_count >= self->fault_after_frames) {
    self->fault_injected = TRUE;
    const gchar* gate = g_getenv("SQUEAKVIEW_ENABLE_FAILURE_INJECTION");
    if (g_strcmp0(gate, "1") != 0) {
      GST_ELEMENT_ERROR(
          self,
          CORE,
          FAILED,
          ("FLIR failure injection was configured without the explicit safety gate"),
          ("set SQUEAKVIEW_ENABLE_FAILURE_INJECTION=1 only for qualification"));
      return GST_FLOW_ERROR;
    }
    if (g_strcmp0(self->fault_kind, "source_read") == 0) {
      GST_ELEMENT_ERROR(
          self,
          RESOURCE,
          READ,
          ("Injected FLIR source read failure"),
          ("after-frames=%" G_GUINT64_FORMAT, self->fault_after_frames));
    } else if (g_strcmp0(self->fault_kind, "source_incomplete") == 0) {
      GST_ELEMENT_ERROR(
          self,
          STREAM,
          FAILED,
          ("Injected FLIR incomplete-frame failure"),
          ("after-frames=%" G_GUINT64_FORMAT, self->fault_after_frames));
    } else if (g_strcmp0(self->fault_kind, "capture_ledger_write") == 0) {
      GST_ELEMENT_ERROR(
          self,
          RESOURCE,
          WRITE,
          ("Injected FLIR capture-ledger write failure"),
          ("after-frames=%" G_GUINT64_FORMAT, self->fault_after_frames));
    } else {
      GST_ELEMENT_ERROR(
          self,
          CORE,
          FAILED,
          ("Unsupported FLIR failure-injection kind"),
          ("kind=%s", self->fault_kind ? self->fault_kind : ""));
    }
    return GST_FLOW_ERROR;
  }

  while (!g_atomic_int_get(&self->stopping)) {
    ImagePtr image;
    try {
      image = (*self->camera)->GetNextImage(self->timeout_ms);
    } catch (const Spinnaker::Exception& exc) {
      const std::string message = spinnaker_exception_message(exc);
      if (exception_is_timeout(exc)) {
        ++self->total_timeouts;
        ++self->consecutive_timeouts;
        if (self->max_consecutive_timeouts == 0 ||
            self->consecutive_timeouts < self->max_consecutive_timeouts) {
          continue;
        }
        GST_ELEMENT_ERROR(
            self,
            RESOURCE,
            READ,
            ("FLIR camera exceeded consecutive timeout limit"),
            ("serial=%s consecutive=%u total=%" G_GUINT64_FORMAT " timeout-ms=%u: %s",
             self->resolved_serial ? self->resolved_serial : "",
             self->consecutive_timeouts,
             self->total_timeouts,
             self->timeout_ms,
             message.c_str()));
      } else {
        GST_ELEMENT_ERROR(
            self,
            RESOURCE,
            READ,
            ("Failed to read FLIR Spinnaker frame"),
            ("%s", message.c_str()));
      }
      return GST_FLOW_ERROR;
    }

    if (!image) {
      continue;
    }
    self->consecutive_timeouts = 0;
    const guint64 host_received_monotonic_ns = static_cast<guint64>(g_get_monotonic_time()) * 1000;
    const guint64 host_received_unix_ns = static_cast<guint64>(g_get_real_time()) * 1000;

    if (image->IsIncomplete()) {
      const auto status = image->GetImageStatus();
      ++self->total_incomplete;
      GST_WARNING_OBJECT(self, "incomplete frame: %s", Spinnaker::Image::GetImageStatusDescription(status));
      gst_element_post_message(
          GST_ELEMENT(self),
          gst_message_new_element(
              GST_OBJECT(self),
              gst_structure_new(
                  "flir-incomplete-frame",
                  "camera-index",
                  G_TYPE_UINT,
                  self->camera_index,
                  "camera-serial",
                  G_TYPE_STRING,
                  self->resolved_serial ? self->resolved_serial : "",
                  "host-monotonic-ns",
                  G_TYPE_UINT64,
                  host_received_monotonic_ns,
                  "host-unix-ns",
                  G_TYPE_UINT64,
                  host_received_unix_ns,
                  "status",
                  G_TYPE_STRING,
                  Spinnaker::Image::GetImageStatusDescription(status),
                  nullptr)));
      if (!write_diagnostic_event(
              self,
              "incomplete_frame",
              host_received_unix_ns,
              host_received_monotonic_ns,
              "",
              "",
              std::string("{\"status\":\"") +
                  json_escape(Spinnaker::Image::GetImageStatusDescription(status)) +
                  "\"}")) {
        GST_ELEMENT_ERROR(
            self, RESOURCE, WRITE, ("Failed to write camera diagnostic event"),
            ("path=%s", self->error_log_path ? self->error_log_path : ""));
        image->Release();
        return GST_FLOW_ERROR;
      }
      image->Release();
      if (self->drop_incomplete) {
        continue;
      }
      return GST_FLOW_ERROR;
    }

    GstFlowReturn flow = copy_image_to_buffer(
        self,
        image,
        host_received_monotonic_ns,
        host_received_unix_ns,
        out_buffer);
    image->Release();
    return flow;
  }

  return GST_FLOW_FLUSHING;
}

static void gst_flir_spin_src_class_init(GstFlirSpinSrcClass* klass) {
  GObjectClass* object_class = G_OBJECT_CLASS(klass);
  GstElementClass* element_class = GST_ELEMENT_CLASS(klass);
  GstBaseSrcClass* base_src_class = GST_BASE_SRC_CLASS(klass);
  GstPushSrcClass* push_src_class = GST_PUSH_SRC_CLASS(klass);

  object_class->set_property = gst_flir_spin_src_set_property;
  object_class->get_property = gst_flir_spin_src_get_property;
  object_class->finalize = gst_flir_spin_src_finalize;

  g_object_class_install_property(
      object_class,
      PROP_CAMERA_INDEX,
      g_param_spec_uint(
          "camera-index",
          "Camera index",
          "Index in the Spinnaker camera list",
          0,
          G_MAXUINT,
          0,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_WIDTH,
      g_param_spec_uint(
          "width",
          "Width",
          "Requested camera width and output caps width",
          1,
          G_MAXUINT,
          DEFAULT_WIDTH,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_HEIGHT,
      g_param_spec_uint(
          "height",
          "Height",
          "Requested camera height and output caps height",
          1,
          G_MAXUINT,
          DEFAULT_HEIGHT,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_FPS,
      g_param_spec_uint(
          "fps",
          "FPS",
          "Requested acquisition frame rate and output caps frame rate",
          1,
          G_MAXUINT,
          DEFAULT_FPS,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_PIXEL_FORMAT,
      g_param_spec_string(
          "pixel-format",
          "Pixel format",
          "Spinnaker camera pixel format to request; output is converted to GRAY8",
          "Mono8",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_TRIGGER,
      g_param_spec_boolean(
          "trigger",
          "Trigger",
          "Enable external FrameStart trigger on Line0",
          FALSE,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_TRIGGER_ACTIVATION,
      g_param_spec_string(
          "trigger-activation",
          "Trigger activation",
          "Trigger activation: rising or falling",
          "rising",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_EXPOSURE_US,
      g_param_spec_double(
          "exposure-us",
          "Exposure time",
          "Manual exposure time in microseconds; set to -1 to leave camera default/auto",
          -1.0,
          DBL_MAX,
          10000.0,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_GAIN,
      g_param_spec_double(
          "gain",
          "Gain",
          "Manual gain; set to -1 to leave camera default/auto",
          -1.0,
          DBL_MAX,
          -1.0,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_TIMEOUT_MS,
      g_param_spec_uint(
          "timeout-ms",
          "Timeout",
          "Spinnaker GetNextImage timeout in milliseconds",
          1,
          G_MAXUINT,
          DEFAULT_TIMEOUT_MS,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_DROP_INCOMPLETE,
      g_param_spec_boolean(
          "drop-incomplete",
          "Drop incomplete",
          "Drop incomplete camera frames instead of erroring. Scientific acquisition should leave this false.",
          FALSE,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_BUFFER_HANDLING,
      g_param_spec_string(
          "buffer-handling",
          "Stream buffer handling",
          "Optional Spinnaker StreamBufferHandlingMode, e.g. NewestOnly, OldestFirst",
          "OldestFirst",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_STREAM_BUFFER_COUNT,
      g_param_spec_uint(
          "stream-buffer-count",
          "Stream buffer count",
          "Requested Spinnaker host transport buffers; zero leaves the camera default",
          0,
          G_MAXUINT,
          64,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_CAMERA_SERIAL,
      g_param_spec_string(
          "camera-serial",
          "Camera serial",
          "Stable Spinnaker camera serial number; empty uses camera-index",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_METADATA_PROFILE,
      g_param_spec_string(
          "metadata-profile",
          "Metadata profile",
          "Chunk metadata profile: scientific or off",
          DEFAULT_METADATA_PROFILE,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_MAX_CONSECUTIVE_TIMEOUTS,
      g_param_spec_uint(
          "max-consecutive-timeouts",
          "Maximum consecutive timeouts",
          "Fail acquisition after this many consecutive GetNextImage timeouts; zero retries indefinitely",
          0,
          G_MAXUINT,
          DEFAULT_MAX_CONSECUTIVE_TIMEOUTS,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_CAPTURE_LOG_PATH,
      g_param_spec_string(
          "capture-log-path",
          "Capture log path",
          "Line-delimited JSON ledger written for every source buffer before downstream delivery",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_FRAME_MANIFEST_PATH,
      g_param_spec_string(
          "frame-manifest-path",
          "Frame manifest path",
          "Canonical CSV frame index written for every source buffer before downstream delivery",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_CAMERA_TELEMETRY_PATH,
      g_param_spec_string(
          "camera-telemetry-path",
          "Camera telemetry path",
          "Low-rate CSV camera and transport telemetry written live during acquisition",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_ERROR_LOG_PATH,
      g_param_spec_string(
          "error-log-path",
          "Camera error log path",
          "CSV camera integrity events written live during acquisition",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_CAMERA_RUNTIME_PATH,
      g_param_spec_string(
          "camera-runtime-path",
          "Camera runtime metadata path",
          "JSON snapshot of the resolved camera identity and acquisition configuration",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_FAULT_AFTER_FRAMES,
      g_param_spec_uint64(
          "fault-after-frames",
          "Qualification fault frame",
          "Inject the selected qualification-only failure after this many emitted frames; zero disables",
          0,
          G_MAXUINT64,
          0,
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));
  g_object_class_install_property(
      object_class,
      PROP_FAULT_KIND,
      g_param_spec_string(
          "fault-kind",
          "Qualification fault kind",
          "Qualification-only failure kind; requires the explicit environment safety gate",
          "",
          static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  gst_element_class_set_static_metadata(
      element_class,
      "FLIR Spinnaker Source",
      "Source/Video",
      "Captures FLIR/Teledyne camera frames through the Spinnaker C++ SDK",
      "SqueakView");
  gst_element_class_add_static_pad_template(element_class, &src_template);

  base_src_class->start = GST_DEBUG_FUNCPTR(gst_flir_spin_src_start);
  base_src_class->stop = GST_DEBUG_FUNCPTR(gst_flir_spin_src_stop);
  base_src_class->unlock = GST_DEBUG_FUNCPTR(gst_flir_spin_src_unlock);
  base_src_class->unlock_stop = GST_DEBUG_FUNCPTR(gst_flir_spin_src_unlock_stop);
  base_src_class->get_caps = GST_DEBUG_FUNCPTR(gst_flir_spin_src_get_caps);
  push_src_class->create = GST_DEBUG_FUNCPTR(gst_flir_spin_src_create);
}

static void gst_flir_spin_src_init(GstFlirSpinSrc* self) {
  self->camera_index = 0;
  self->width = DEFAULT_WIDTH;
  self->height = DEFAULT_HEIGHT;
  self->fps = DEFAULT_FPS;
  self->pixel_format = g_strdup("Mono8");
  self->trigger = FALSE;
  self->trigger_activation = g_strdup("rising");
  self->exposure_us = 10000.0;
  self->gain = -1.0;
  self->timeout_ms = DEFAULT_TIMEOUT_MS;
  self->drop_incomplete = FALSE;
  self->buffer_handling = g_strdup("OldestFirst");
  self->stream_buffer_count = 64;
  self->camera_serial = g_strdup("");
  self->metadata_profile = g_strdup(DEFAULT_METADATA_PROFILE);
  self->max_consecutive_timeouts = DEFAULT_MAX_CONSECUTIVE_TIMEOUTS;
  self->capture_log_path = g_strdup("");
  self->capture_log = nullptr;
  self->frame_manifest_path = g_strdup("");
  self->frame_manifest = nullptr;
  self->camera_telemetry_path = g_strdup("");
  self->camera_telemetry = nullptr;
  self->error_log_path = g_strdup("");
  self->error_log = nullptr;
  self->camera_runtime_path = g_strdup("");
  self->fault_after_frames = 0;
  self->fault_kind = g_strdup("");
  self->fault_injected = FALSE;
  self->latency_reference_enabled = FALSE;
  self->actual_width = 0;
  self->actual_height = 0;
  self->actual_fps = 0.0;
  self->actual_exposure_us = -1.0;
  self->actual_gain_db = -1.0;
  self->frame_count = 0;
  self->camera_timestamp_base = 0;
  self->last_frame_id = 0;
  self->last_stream_frame_id = 0;
  self->have_last_stream_frame_id = FALSE;
  self->start_time = GST_CLOCK_TIME_NONE;
  self->frame_duration = gst_util_uint64_scale_int(GST_SECOND, 1, DEFAULT_FPS);
  self->have_camera_timestamp_base = FALSE;
  self->have_last_frame_id = FALSE;
  g_atomic_int_set(&self->stopping, FALSE);
  self->acquisition_started = FALSE;
  self->chunks_enabled = FALSE;
  self->consecutive_timeouts = 0;
  self->total_timeouts = 0;
  self->total_incomplete = 0;
  self->total_frame_gaps = 0;
  self->total_crc_failures = 0;
  self->actual_stream_buffer_count = 0;
  self->resolved_serial = g_strdup("");
  self->device_model = g_strdup("");
  self->firmware_version = g_strdup("");
  self->actual_pixel_format = g_strdup("");
  self->enabled_chunks = g_strdup("");
  self->timestamp_increment_ns = 0;
  self->have_timestamp_latch = FALSE;
  self->timestamp_latch_value = 0;
  self->latch_host_monotonic_before_ns = 0;
  self->latch_host_monotonic_after_ns = 0;
  self->latch_host_unix_before_ns = 0;
  self->latch_host_unix_after_ns = 0;
  self->last_telemetry_frame = G_MAXUINT64;
  self->system = nullptr;
  self->camera_list = nullptr;
  self->camera = nullptr;
  self->processor = nullptr;

  gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
  gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
}

static gboolean plugin_init(GstPlugin* plugin) {
  return gst_element_register(plugin, "flirspinsrc", GST_RANK_NONE, GST_TYPE_FLIR_SPIN_SRC);
}

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    flirspinsrc,
    "FLIR Spinnaker camera source",
    plugin_init,
    "0.2.0",
    "LGPL",
    "SqueakView",
    "https://github.com")
