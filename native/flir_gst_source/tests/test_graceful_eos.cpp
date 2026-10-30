// Include the production translation unit so these camera-free tests exercise
// its actual GstElement event dispatch and create() EOS boundary.
#include "../src/gstflirspinsrc.cpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

struct DrainHarness {
  std::mutex mutex;
  std::condition_variable changed;
  bool reading = false;
  bool release = false;
  bool idle = false;
  unsigned created = 0;
  std::atomic<unsigned> delivered{0};
};

struct DrainTestSrc {
  GstFlirSpinSrc parent;
  DrainHarness* harness;
};
struct DrainTestSrcClass { GstFlirSpinSrcClass parent; };
G_DEFINE_TYPE(DrainTestSrc, drain_test_src, GST_TYPE_FLIR_SPIN_SRC)

static gboolean test_start(GstBaseSrc* base) {
  auto* self = reinterpret_cast<DrainTestSrc*>(base);
  g_atomic_int_set(&self->parent.stopping, FALSE);
  g_atomic_int_set(&self->parent.graceful_eos_requested, FALSE);
  self->parent.acquisition_started = TRUE;
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
      "GRAY8", "width", G_TYPE_INT, 1, "height", G_TYPE_INT, 1,
      "framerate", GST_TYPE_FRACTION, 30, 1, nullptr);
  gboolean result = gst_base_src_set_caps(base, caps);
  gst_caps_unref(caps);
  return result;
}

static gboolean test_stop(GstBaseSrc* base) {
  auto* self = reinterpret_cast<DrainTestSrc*>(base);
  self->parent.acquisition_started = FALSE;
  return TRUE;
}

static GstFlowReturn test_create(GstPushSrc* push, GstBuffer** buffer) {
  auto* self = reinterpret_cast<DrainTestSrc*>(push);
  auto* h = self->harness;
  // This is the production source's next-read boundary. In particular, there
  // is deliberately no EOS check after the fake blocking camera read succeeds.
  if (g_atomic_int_get(&self->parent.graceful_eos_requested)) return GST_FLOW_EOS;
  const unsigned index = h->created;
  if (index == 1 || h->idle) {
    std::unique_lock<std::mutex> lock(h->mutex);
    h->reading = true;
    h->changed.notify_all();
    const bool released = h->changed.wait_for(lock, std::chrono::seconds(3), [&] { return h->release; });
    if (!released) return GST_FLOW_ERROR;
    // An idle triggered camera times out rather than acquiring another frame.
    if (h->idle && g_atomic_int_get(&self->parent.graceful_eos_requested)) return GST_FLOW_EOS;
  }
  *buffer = gst_buffer_new_allocate(nullptr, 1, nullptr);
  GST_BUFFER_PTS(*buffer) = index * GST_SECOND / 30;
  GST_BUFFER_DURATION(*buffer) = GST_SECOND / 30;
  ++h->created;
  return GST_FLOW_OK;
}

static void test_finalize(GObject* object) {
  auto* self = reinterpret_cast<DrainTestSrc*>(object);
  delete self->harness;
  G_OBJECT_CLASS(drain_test_src_parent_class)->finalize(object);
}

static void drain_test_src_class_init(DrainTestSrcClass* klass) {
  // Inherit the actual production send_event, unlock and unlock_stop handlers.
  // A test-only baseline switch restores GstBaseSrc's old external EOS behavior
  // to demonstrate the same final-buffer loss with the identical blocked read.
  if (g_getenv("SQUEAKVIEW_TEST_BASESRC_EOS")) {
    GST_ELEMENT_CLASS(klass)->send_event =
        GST_ELEMENT_CLASS(gst_flir_spin_src_parent_class)->send_event;
  }
  auto* base = GST_BASE_SRC_CLASS(klass);
  base->start = test_start;
  base->stop = test_stop;
  GST_PUSH_SRC_CLASS(klass)->create = test_create;
  G_OBJECT_CLASS(klass)->finalize = test_finalize;
}
static void drain_test_src_init(DrainTestSrc* self) { self->harness = new DrainHarness; }

static void handoff(GstElement*, GstBuffer*, GstPad*, gpointer user) {
  ++static_cast<DrainHarness*>(user)->delivered;
}

static void assert_pipeline_drains(unsigned sources, bool idle) {
  GstElement* pipeline = gst_pipeline_new(nullptr);
  std::vector<DrainTestSrc*> cameras;
  for (unsigned index = 0; index < sources; ++index) {
    auto* src = reinterpret_cast<DrainTestSrc*>(g_object_new(drain_test_src_get_type(), nullptr));
    src->harness->idle = idle;
    cameras.push_back(src);
    GstElement* queue = gst_element_factory_make("queue", nullptr);
    GstElement* sink = gst_element_factory_make("fakesink", nullptr);
    g_object_set(sink, "sync", FALSE, "async", FALSE, "signal-handoffs", TRUE, nullptr);
    g_signal_connect(sink, "handoff", G_CALLBACK(handoff), src->harness);
    gst_bin_add_many(GST_BIN(pipeline), GST_ELEMENT(src), queue, sink, nullptr);
    g_assert_true(gst_element_link_many(GST_ELEMENT(src), queue, sink, nullptr));
  }
  g_assert_cmpint(gst_element_set_state(pipeline, GST_STATE_PLAYING), !=, GST_STATE_CHANGE_FAILURE);
  for (auto* src : cameras) {
    auto* h = src->harness;
    std::unique_lock<std::mutex> lock(h->mutex);
    g_assert_true(h->changed.wait_for(lock, std::chrono::seconds(3), [&] { return h->reading; }));
  }
  // A real camera read completes independently of send_event(). GstBaseSrc's
  // default EOS handler may wait for the streaming thread, so release our fake
  // read from another thread once the EOS handler has requested either drain
  // (fixed path) or unlock/flush (old path).
  std::thread complete_camera_read([&] {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    for (auto* src : cameras) {
      while (!g_atomic_int_get(&src->parent.graceful_eos_requested) &&
             !g_atomic_int_get(&src->parent.stopping) &&
             std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      auto* h = src->harness;
      std::lock_guard<std::mutex> lock(h->mutex);
      h->release = true;
      h->changed.notify_all();
    }
  });
  // Service Maker Pipeline.stop() uses precisely this external EOS entry point.
  g_assert_true(gst_element_send_event(pipeline, gst_event_new_eos()));
  // Repeated operator stop/EOS must not flush or duplicate the terminal event.
  if (!g_getenv("SQUEAKVIEW_TEST_BASESRC_EOS")) {
    g_assert_true(gst_element_send_event(pipeline, gst_event_new_eos()));
  }
  for (auto* src : cameras) {
    if (!g_getenv("SQUEAKVIEW_TEST_BASESRC_EOS")) {
      g_assert_true(g_atomic_int_get(&src->parent.graceful_eos_requested));
      g_assert_false(g_atomic_int_get(&src->parent.stopping));
    }
  }
  complete_camera_read.join();
  GstBus* bus = gst_element_get_bus(pipeline);
  GstMessage* message = gst_bus_timed_pop_filtered(bus, 5 * GST_SECOND,
      static_cast<GstMessageType>(GST_MESSAGE_EOS | GST_MESSAGE_ERROR));
  g_assert_nonnull(message);
  g_assert_cmpint(GST_MESSAGE_TYPE(message), ==, GST_MESSAGE_EOS);
  for (auto* src : cameras) {
    const unsigned expected = idle ? 0 : 2;
    g_assert_cmpuint(src->harness->created, ==, expected);
    g_assert_cmpuint(src->harness->delivered.load(), ==, expected);
  }
  gst_message_unref(message);
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
}

static void production_eos_boundary() {
  auto* src = GST_FLIR_SPIN_SRC(g_object_new(GST_TYPE_FLIR_SPIN_SRC, nullptr));
  GstBuffer* buffer = nullptr;
  // No camera acquisition is started: this directly exercises the production
  // event virtual and create boundary without touching a device or opening files.
  g_assert_true(gst_element_send_event(GST_ELEMENT(src), gst_event_new_eos()));
  g_assert_true(g_atomic_int_get(&src->graceful_eos_requested));
  g_assert_false(g_atomic_int_get(&src->stopping));
  g_assert_cmpint(gst_flir_spin_src_create(GST_PUSH_SRC(src), &buffer), ==, GST_FLOW_EOS);
  g_assert_null(buffer);
  g_assert_cmpuint(src->frame_count, ==, 0);
  gst_object_unref(src);
}

static void native_timestamp_contract() {
  g_assert_false(native_timestamp_is_valid(std::nullopt, 0, 0));
  g_assert_true(native_timestamp_is_valid(0, 0, 0));
  g_assert_true(native_timestamp_is_valid(101, 1, 100));
  g_assert_false(native_timestamp_is_valid(100, 1, 100));
  g_assert_false(native_timestamp_is_valid(99, 1, 100));
  auto* src = GST_FLIR_SPIN_SRC(g_object_new(GST_TYPE_FLIR_SPIN_SRC, nullptr));
  gboolean available = FALSE;
  g_object_get(src, "native-timing-validation", &available, nullptr);
  g_assert_true(available);
  gst_object_unref(src);
}

static void compact_capture_health_contract() {
  auto* src = GST_FLIR_SPIN_SRC(g_object_new(GST_TYPE_FLIR_SPIN_SRC, nullptr));
  gchar* directory = g_dir_make_tmp("squeakview-health-XXXXXX", nullptr);
  g_assert_nonnull(directory);
  gchar* path = g_build_filename(directory, "capture_health_cam0.json", nullptr);
  g_object_set(src, "capture-health-path", path, nullptr);
  src->health_active = TRUE;
  src->frame_manifest = std::tmpfile();
  g_assert_nonnull(src->frame_manifest);
  std::fputs("canonical frame row\n", src->frame_manifest);
  g_assert_true(write_capture_health(src, false));
  gchar* data = nullptr;
  gsize length = 0;
  g_assert_true(g_file_get_contents(path, &data, &length, nullptr));
  g_assert_nonnull(std::strstr(data, "\"state\":\"recording\""));
  g_assert_nonnull(std::strstr(data, "\"source_frames\":0"));
  g_free(data);
  // A whole day's count still occupies one bounded object, not a ledger.
  src->frame_count = 2592000;
  src->total_crc_failures = 1;
  close_camera(src);
  g_assert_true(g_file_get_contents(path, &data, &length, nullptr));
  g_assert_cmpuint(length, <, 1024);
  g_assert_nonnull(std::strstr(data, "\"state\":\"closed\""));
  g_assert_nonnull(std::strstr(data, "\"source_frames\":2592000"));
  g_assert_nonnull(std::strstr(data, "\"total_crc_failures\":1"));
  g_assert_nonnull(std::strstr(data, "\"native_timing_validated\":true"));
  g_assert_null(std::strstr(data, "camera_timestamp_ns"));
  g_free(data);
  // Failed checkpoint publication is reported; no partial file is published.
  g_object_set(src, "capture-health-path", "/nonexistent-squeakview-health/health.json", nullptr);
  src->health_active = TRUE;
  g_assert_false(write_capture_health(src, false));
  src->health_active = FALSE;
  gst_object_unref(src);
  std::remove(path);
  std::remove(directory);
  g_free(path);
  g_free(directory);
}

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  g_test_init(&argc, &argv, nullptr);
  g_test_add_func("/flir/timing/live-native-validation", native_timestamp_contract);
  g_test_add_func("/flir/health/compact-checkpoint-and-close", compact_capture_health_contract);
  g_test_add_func("/flir/eos/production-boundary", production_eos_boundary);
  g_test_add_func("/flir/eos/inflight-free-running", [] { assert_pipeline_drains(1, false); });
  g_test_add_func("/flir/eos/inflight-two-cameras", [] { assert_pipeline_drains(2, false); });
  g_test_add_func("/flir/eos/idle-triggered", [] { assert_pipeline_drains(1, true); });
  return g_test_run();
}
