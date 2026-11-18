// Included after attach_flir_frame_meta: the optional process receives exactly
// the source identity/timestamps, even when it skips replaceable live frames.
struct GstSqueakLiveSrc { GstPushSrc parent; gchar* path; guint width, height, fps;
  gint stopping; guint64 sequence; squeakview::LiveMailbox* mailbox; };
struct GstSqueakLiveSrcClass { GstPushSrcClass parent; };
G_DEFINE_TYPE(GstSqueakLiveSrc, gst_squeak_live_src, GST_TYPE_PUSH_SRC)

static void live_set(GObject* object, guint id, const GValue* value, GParamSpec*) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(object);
  switch (id) {
    case 1: g_free(self->path); self->path = g_value_dup_string(value); break;
    case 2: self->width = g_value_get_uint(value); break;
    case 3: self->height = g_value_get_uint(value); break;
    case 4: self->fps = g_value_get_uint(value); break;
    case 5: if (g_value_get_boolean(value)) g_atomic_int_set(&self->stopping, TRUE); break;
  }
}
static void live_get(GObject* object, guint id, GValue* value, GParamSpec*) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(object);
  switch (id) {
    case 1: g_value_set_string(value, self->path); break;
    case 2: g_value_set_uint(value, self->width); break;
    case 3: g_value_set_uint(value, self->height); break;
    case 4: g_value_set_uint(value, self->fps); break;
    case 5: g_value_set_boolean(value, g_atomic_int_get(&self->stopping)); break;
  }
}
static GstCaps* live_get_caps(GstBaseSrc* base, GstCaps* filter) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(base);
  // BaseSrc negotiates again after start(). Advertise the actual mailbox
  // geometry so it cannot fixate the broad pad template to 1x1 at 1 FPS.
  auto* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "GRAY8",
      "width", G_TYPE_INT, self->width, "height", G_TYPE_INT, self->height,
      "framerate", GST_TYPE_FRACTION, self->fps, 1, nullptr);
  if (!filter) return caps;
  auto* intersected = gst_caps_intersect_full(filter, caps, GST_CAPS_INTERSECT_FIRST);
  gst_caps_unref(caps);
  return intersected;
}
static gboolean live_start(GstBaseSrc* base) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(base);
  g_atomic_int_set(&self->stopping, FALSE);
  self->sequence = 0;
  try { self->mailbox = new squeakview::LiveMailbox(self->path); }
  catch (const std::exception& exc) {
    GST_ELEMENT_ERROR(self, RESOURCE, OPEN_READ, ("Cannot open live mailbox"), ("%s", exc.what()));
    return FALSE;
  }
  return TRUE;
}
static gboolean live_stop(GstBaseSrc* base) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(base);
  delete self->mailbox; self->mailbox = nullptr;
  return TRUE;
}
static gboolean live_unlock(GstBaseSrc* base) {
  g_atomic_int_set(&reinterpret_cast<GstSqueakLiveSrc*>(base)->stopping, TRUE);
  return TRUE;
}
static GstFlowReturn live_create(GstPushSrc* push, GstBuffer** output) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(push);
  while (!g_atomic_int_get(&self->stopping)) {
    squeakview::LiveFrame frame;
    bool closed = false;
    bool available = false;
    try { available = self->mailbox && self->mailbox->read(self->sequence, frame, closed); }
    catch (const std::exception& exc) {
      GST_ELEMENT_ERROR(self, RESOURCE, READ, ("Cannot read live frame"), ("%s", exc.what()));
      return GST_FLOW_ERROR;
    }
    if (available) {
      if (frame.header.width != self->width || frame.header.height != self->height) return GST_FLOW_ERROR;
      GstBuffer* buffer = gst_buffer_new_allocate(nullptr, frame.pixels.size(), nullptr);
      if (!buffer) return GST_FLOW_ERROR;
      gst_buffer_fill(buffer, 0, frame.pixels.data(), frame.pixels.size());
      GST_BUFFER_PTS(buffer) = frame.header.pts;
      GST_BUFFER_DTS(buffer) = frame.header.pts;
      GST_BUFFER_DURATION(buffer) = frame.header.duration;
      if (!attach_flir_frame_meta(buffer, frame.metadata)) { gst_buffer_unref(buffer); return GST_FLOW_ERROR; }
      self->sequence = frame.header.sequence;
      *output = buffer;
      return GST_FLOW_OK;
    }
    if (closed) return GST_FLOW_EOS;
    g_usleep(5000);
  }
  return GST_FLOW_EOS;
}
static void live_finalize(GObject* object) {
  auto* self = reinterpret_cast<GstSqueakLiveSrc*>(object);
  delete self->mailbox; g_free(self->path);
  G_OBJECT_CLASS(gst_squeak_live_src_parent_class)->finalize(object);
}
static void gst_squeak_live_src_class_init(GstSqueakLiveSrcClass* klass) {
  auto* object = G_OBJECT_CLASS(klass);
  object->set_property = live_set; object->get_property = live_get; object->finalize = live_finalize;
  auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS);
  g_object_class_install_property(object, 1, g_param_spec_string("mailbox-path", "Mailbox", "Private latest-frame mailbox", "", flags));
  g_object_class_install_property(object, 2, g_param_spec_uint("width", "Width", "Width", 1, 8192, 1440, flags));
  g_object_class_install_property(object, 3, g_param_spec_uint("height", "Height", "Height", 1, 8192, 1080, flags));
  g_object_class_install_property(object, 4, g_param_spec_uint("fps", "FPS", "Nominal FPS; native PTS preserved", 1, 1000, 30, flags));
  g_object_class_install_property(object, 5, g_param_spec_boolean("graceful-eos", "EOS", "Nonblocking stop", FALSE,
      static_cast<GParamFlags>(flags | GST_PARAM_MUTABLE_PLAYING)));
  auto* base = GST_BASE_SRC_CLASS(klass);
  base->get_caps = live_get_caps;
  base->start = live_start; base->stop = live_stop; base->unlock = live_unlock;
  GST_PUSH_SRC_CLASS(klass)->create = live_create;
  auto* element = GST_ELEMENT_CLASS(klass);
  gst_element_class_set_static_metadata(element, "SqueakView optional live source", "Source/Video", "Read replaceable live frames", "SqueakView");
  auto* caps = gst_caps_from_string("video/x-raw,format=GRAY8,width=[1,8192],height=[1,8192],framerate=[1/1,1000/1]");
  gst_element_class_add_pad_template(element, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);
}
static void gst_squeak_live_src_init(GstSqueakLiveSrc* self) {
  self->path = g_strdup(""); self->width = 1440; self->height = 1080; self->fps = 30;
  self->stopping = FALSE; self->sequence = 0; self->mailbox = nullptr;
  gst_base_src_set_live(GST_BASE_SRC(self), TRUE);
  gst_base_src_set_format(GST_BASE_SRC(self), GST_FORMAT_TIME);
}
