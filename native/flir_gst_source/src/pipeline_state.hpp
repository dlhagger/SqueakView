#pragma once
#include <gst/gst.h>

// Service Maker's wait() does not throw when its NULL transition fails. Expose
// a non-waiting state query through the sources we already own, without relying
// on private Python/C++ pointer layouts or SDK-version-specific bindings.
static gboolean owning_pipeline_is_null(GstElement* element) {
  GstObject* owner = GST_OBJECT(gst_object_ref(element));
  while (owner && !GST_IS_PIPELINE(owner)) {
    GstObject* parent = gst_object_get_parent(owner);
    gst_object_unref(owner);
    owner = parent;
  }
  if (!owner) return FALSE;
  GstState current = GST_STATE_VOID_PENDING;
  GstState pending = GST_STATE_VOID_PENDING;
  const auto result = gst_element_get_state(GST_ELEMENT(owner), &current, &pending, 0);
  gst_object_unref(owner);
  return result == GST_STATE_CHANGE_SUCCESS && current == GST_STATE_NULL &&
         pending == GST_STATE_VOID_PENDING;
}
