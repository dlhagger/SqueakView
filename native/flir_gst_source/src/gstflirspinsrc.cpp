#include <gst/base/gstpushsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <stdexcept>

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using Spinnaker::CameraList;
using Spinnaker::CameraPtr;
using Spinnaker::ImagePtr;
using Spinnaker::ImageProcessor;
using Spinnaker::PixelFormat_Mono8;
using Spinnaker::System;
using Spinnaker::SystemPtr;
using Spinnaker::GenApi::CBooleanPtr;
using Spinnaker::GenApi::CEnumEntryPtr;
using Spinnaker::GenApi::CEnumerationPtr;
using Spinnaker::GenApi::CFloatPtr;
using Spinnaker::GenApi::CIntegerPtr;
using Spinnaker::GenApi::INodeMap;
using Spinnaker::GenApi::IsAvailable;
using Spinnaker::GenApi::IsReadable;
using Spinnaker::GenApi::IsWritable;

namespace {

constexpr guint DEFAULT_WIDTH = 1440;
constexpr guint DEFAULT_HEIGHT = 1080;
constexpr guint DEFAULT_FPS = 30;
constexpr guint DEFAULT_TIMEOUT_MS = 1000;

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

  guint actual_width;
  guint actual_height;
  guint actual_fps;
  guint64 frame_count;
  guint64 camera_timestamp_base;
  guint64 last_frame_id;
  GstClockTime start_time;
  GstClockTime frame_duration;
  gboolean have_camera_timestamp_base;
  gboolean have_last_frame_id;
  gboolean stopping;
  gboolean acquisition_started;

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

static bool message_is_timeout(const std::string& message) {
  return message.find("Timeout") != std::string::npos || message.find("timeout") != std::string::npos;
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

static void maybe_set_stream_buffer_handling(GstFlirSpinSrc* self, CameraPtr cam) {
  if (!self->buffer_handling || !*self->buffer_handling) {
    return;
  }

  INodeMap& stream_map = cam->GetTLStreamNodeMap();
  enum_set(stream_map, "StreamBufferHandlingMode", self->buffer_handling, false);
}

static void configure_camera(GstFlirSpinSrc* self, CameraPtr cam) {
  INodeMap& node_map = cam->GetNodeMap();

  maybe_set_stream_buffer_handling(self, cam);
  enum_set(node_map, "AcquisitionMode", "Continuous", false);

  if (self->pixel_format && *self->pixel_format) {
    enum_set(node_map, "PixelFormat", self->pixel_format, true);
  }

  CIntegerPtr width_node = node_map.GetNode("Width");
  if (IsAvailable(width_node) && IsWritable(width_node)) {
    const int64_t desired = static_cast<int64_t>(self->width);
    const int64_t clamped = std::min<int64_t>(desired, width_node->GetMax());
    width_node->SetValue(clamped);
  }

  CIntegerPtr height_node = node_map.GetNode("Height");
  if (IsAvailable(height_node) && IsWritable(height_node)) {
    const int64_t desired = static_cast<int64_t>(self->height);
    const int64_t clamped = std::min<int64_t>(desired, height_node->GetMax());
    height_node->SetValue(clamped);
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
      const double clamped = std::max(exposure_node->GetMin(), std::min(self->exposure_us, exposure_node->GetMax()));
      exposure_node->SetValue(clamped);
    }
  }

  if (self->gain >= 0.0) {
    enum_set(node_map, "GainAuto", "Off", false);
    CFloatPtr gain_node = node_map.GetNode("Gain");
    if (IsAvailable(gain_node) && IsWritable(gain_node)) {
      const double clamped = std::max(gain_node->GetMin(), std::min(self->gain, gain_node->GetMax()));
      gain_node->SetValue(clamped);
    }
  }

  maybe_set_trigger(self, cam);

  self->actual_width = self->width;
  self->actual_height = self->height;
  self->actual_fps = self->fps;
  if (IsAvailable(width_node) && IsReadable(width_node)) {
    self->actual_width = static_cast<guint>(width_node->GetValue());
  }
  if (IsAvailable(height_node) && IsReadable(height_node)) {
    self->actual_height = static_cast<guint>(height_node->GetValue());
  }
  if (IsAvailable(fps_node) && IsReadable(fps_node) && !self->trigger) {
    self->actual_fps = std::max<guint>(1, static_cast<guint>(fps_node->GetValue() + 0.5));
  }
}

static GstCaps* make_caps(GstFlirSpinSrc* self) {
  const guint width = self->actual_width ? self->actual_width : self->width;
  const guint height = self->actual_height ? self->actual_height : self->height;
  const guint fps = self->actual_fps ? self->actual_fps : self->fps;
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
  self->stopping = TRUE;

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
}

static gboolean open_camera(GstFlirSpinSrc* self, std::string& error) {
  try {
    self->system = new SystemPtr(System::GetInstance());
    self->camera_list = new CameraList((*self->system)->GetCameras());

    const unsigned int count = self->camera_list->GetSize();
    if (count == 0) {
      error = "No Spinnaker cameras found";
      close_camera(self);
      return FALSE;
    }
    if (self->camera_index >= count) {
      std::ostringstream oss;
      oss << "camera-index " << self->camera_index << " out of range; found " << count << " camera(s)";
      error = oss.str();
      close_camera(self);
      return FALSE;
    }

    CameraPtr cam = self->camera_list->GetByIndex(self->camera_index);
    self->camera = new CameraPtr(cam);
    (*self->camera)->Init();

    configure_camera(self, *self->camera);
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
    self->stopping = FALSE;
    self->frame_count = 0;
    self->camera_timestamp_base = 0;
    self->last_frame_id = 0;
    self->have_camera_timestamp_base = FALSE;
    self->have_last_frame_id = FALSE;
    self->start_time = gst_util_get_timestamp();
    self->frame_duration = self->actual_fps > 0 ? gst_util_uint64_scale_int(GST_SECOND, 1, self->actual_fps) : GST_CLOCK_TIME_NONE;

    GST_INFO_OBJECT(
        self,
        "started FLIR camera index=%u format=%s output=GRAY8 %ux%u@%u trigger=%s",
        self->camera_index,
        self->pixel_format ? self->pixel_format : "",
        self->actual_width,
        self->actual_height,
        self->actual_fps,
        self->trigger ? "true" : "false");
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

static GstFlowReturn copy_image_to_buffer(GstFlirSpinSrc* self, const ImagePtr& image, GstBuffer** out_buffer) {
  ImagePtr mono;
  if (image->GetPixelFormat() != PixelFormat_Mono8) {
    mono = self->processor->Convert(image, PixelFormat_Mono8);
  } else {
    mono = image;
  }

  const size_t width = mono->GetWidth();
  const size_t height = mono->GetHeight();
  const size_t stride = mono->GetStride();
  const size_t expected_width = self->actual_width;
  const size_t expected_height = self->actual_height;
  if (width != expected_width || height != expected_height) {
    GST_ELEMENT_ERROR(
        self,
        STREAM,
        FORMAT,
        ("Camera frame dimensions changed"),
        ("Got %zux%zu, expected %zux%zu", width, height, expected_width, expected_height));
    return GST_FLOW_ERROR;
  }

  const size_t payload_size = expected_width * expected_height;
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, payload_size, nullptr);
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
  if (stride == expected_width) {
    std::memcpy(dst, src, payload_size);
  } else {
    for (size_t row = 0; row < expected_height; ++row) {
      std::memcpy(dst + row * expected_width, src + row * stride, expected_width);
    }
  }
  gst_buffer_unmap(buffer, &map);

  guint64 frame_id = self->frame_count;
  gboolean have_frame_id = TRUE;
  try {
    frame_id = static_cast<guint64>(image->GetFrameID());
  } catch (...) {
    have_frame_id = FALSE;
  }

  guint64 camera_timestamp = 0;
  gboolean have_camera_timestamp = TRUE;
  try {
    camera_timestamp = static_cast<guint64>(image->GetTimeStamp());
  } catch (...) {
    have_camera_timestamp = FALSE;
  }

  if (have_frame_id && self->have_last_frame_id && frame_id != self->last_frame_id + 1) {
    const guint64 expected = self->last_frame_id + 1;
    GST_WARNING_OBJECT(
        self,
        "FLIR frame id gap on camera %u: expected %" G_GUINT64_FORMAT ", got %" G_GUINT64_FORMAT,
        self->camera_index,
        expected,
        frame_id);
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
                frame_id,
                nullptr)));
  }
  if (have_frame_id) {
    self->last_frame_id = frame_id;
    self->have_last_frame_id = TRUE;
  }

  GstClockTime pts = GST_CLOCK_TIME_NONE;
  if (have_camera_timestamp && camera_timestamp > 0) {
    if (!self->have_camera_timestamp_base) {
      self->camera_timestamp_base = camera_timestamp;
      self->have_camera_timestamp_base = TRUE;
    }
    pts = camera_timestamp >= self->camera_timestamp_base ? camera_timestamp - self->camera_timestamp_base : 0;
  }
  if (!GST_CLOCK_TIME_IS_VALID(pts)) {
    const GstClockTime now = gst_util_get_timestamp();
    pts = now >= self->start_time ? now - self->start_time : 0;
  }

  GST_BUFFER_PTS(buffer) = pts;
  GST_BUFFER_DTS(buffer) = GST_BUFFER_PTS(buffer);
  GST_BUFFER_DURATION(buffer) = self->frame_duration;
  GST_BUFFER_OFFSET(buffer) = frame_id;
  GST_BUFFER_OFFSET_END(buffer) = frame_id + 1;
  ++self->frame_count;

  *out_buffer = buffer;
  return GST_FLOW_OK;
}

static GstFlowReturn gst_flir_spin_src_create(GstPushSrc* push_src, GstBuffer** out_buffer) {
  GstFlirSpinSrc* self = GST_FLIR_SPIN_SRC(push_src);
  if (!self->camera || !self->acquisition_started) {
    return GST_FLOW_FLUSHING;
  }

  while (!self->stopping) {
    ImagePtr image;
    try {
      image = (*self->camera)->GetNextImage(self->timeout_ms);
    } catch (const Spinnaker::Exception& exc) {
      const std::string message = spinnaker_exception_message(exc);
      if (message_is_timeout(message)) {
        continue;
      }
      GST_ELEMENT_ERROR(self, RESOURCE, READ, ("Failed to read FLIR Spinnaker frame"), ("%s", message.c_str()));
      return GST_FLOW_ERROR;
    }

    if (!image) {
      continue;
    }

    if (image->IsIncomplete()) {
      const auto status = image->GetImageStatus();
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
                  "status",
                  G_TYPE_STRING,
                  Spinnaker::Image::GetImageStatusDescription(status),
                  nullptr)));
      image->Release();
      if (self->drop_incomplete) {
        continue;
      }
      return GST_FLOW_ERROR;
    }

    GstFlowReturn flow = copy_image_to_buffer(self, image, out_buffer);
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

  gst_element_class_set_static_metadata(
      element_class,
      "FLIR Spinnaker Source",
      "Source/Video",
      "Captures FLIR/Teledyne camera frames through the Spinnaker C++ SDK",
      "SqueakView");
  gst_element_class_add_static_pad_template(element_class, &src_template);

  base_src_class->start = GST_DEBUG_FUNCPTR(gst_flir_spin_src_start);
  base_src_class->stop = GST_DEBUG_FUNCPTR(gst_flir_spin_src_stop);
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
  self->actual_width = 0;
  self->actual_height = 0;
  self->actual_fps = 0;
  self->frame_count = 0;
  self->camera_timestamp_base = 0;
  self->last_frame_id = 0;
  self->start_time = GST_CLOCK_TIME_NONE;
  self->frame_duration = gst_util_uint64_scale_int(GST_SECOND, 1, DEFAULT_FPS);
  self->have_camera_timestamp_base = FALSE;
  self->have_last_frame_id = FALSE;
  self->stopping = FALSE;
  self->acquisition_started = FALSE;
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
    "0.1.0",
    "LGPL",
    "SqueakView",
    "https://github.com")
