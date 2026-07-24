// YOLO26 end-to-end pose parser for the SqueakView DeepStream 9.1 contract.
// Pose keypoints are decoded frame-by-frame from output tensor metadata by PyServiceMaker.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer.h"

struct PoseDet {
  float x1,y1,x2,y2, conf; int cls;
  std::vector<float> kpts;
};

static inline float env_or_default(const char* name, float fallback) {
  if (const char* v = std::getenv(name)) {
    try { return std::stof(v); } catch (...) { return fallback; }
  }
  return fallback;
}

// Undo letterbox padding to map 640x640 net coords back to source frame (e.g., 1440x1080).
static inline void unletterbox(float& x, float& y,
                               float gain, float pad_x, float pad_y,
                               float src_w, float src_h) {
  x = (x - pad_x) / gain;
  y = (y - pad_y) / gain;
  x = std::min(std::max(x, 0.f), src_w - 1.f);
  y = std::min(std::max(y, 0.f), src_h - 1.f);
}
static inline float class_threshold(const NvDsInferParseDetectionParams& params, int cls, float fallback) {
  if (cls >= 0 && static_cast<size_t>(cls) < params.perClassPreclusterThreshold.size()) {
    return params.perClassPreclusterThreshold[cls];
  }
  if (!params.perClassPreclusterThreshold.empty()) {
    return params.perClassPreclusterThreshold[0];
  }
  return fallback;
}

static bool decode_yolo26_pose(const NvDsInferLayerInfo& L,
                               const NvDsInferNetworkInfo& net,
                               const NvDsInferParseDetectionParams& params,
                               std::vector<PoseDet>& dets,
                               std::vector<NvDsInferObjectDetectionInfo>& objects,
                               float conf_thr = 0.25f) {
  if (!L.buffer) return false;
  const float* data = static_cast<const float*>(L.buffer);

  int num_preds = 0;
  int stride = 0;
  bool channel_major = false;
  auto stride_matches = [](int d) -> bool { return d >= 6 && ((d - 6) % 3 == 0); };

  if (L.inferDims.numDims == 2) {
    int d0 = L.inferDims.d[0];
    int d1 = L.inferDims.d[1];
    if (stride_matches(d1)) {
      num_preds = d0;
      stride = d1;
      channel_major = false;
    } else if (stride_matches(d0)) {
      num_preds = d1;
      stride = d0;
      channel_major = true;
    } else {
      std::cout << "[POSE][yolo26] invalid dims=" << d0 << "x" << d1 << " (2D)" << std::endl;
      return false;
    }
  } else if (L.inferDims.numDims == 3) {
    int d0 = L.inferDims.d[0];
    int d1 = L.inferDims.d[1];
    int d2 = L.inferDims.d[2];
    (void)d0;
    if (stride_matches(d2)) {
      num_preds = d1;
      stride = d2;
      channel_major = false;  // [B, N, stride]
    } else if (stride_matches(d1)) {
      num_preds = d2;
      stride = d1;
      channel_major = true;   // [B, stride, N]
    } else {
      std::cout << "[POSE][yolo26] invalid dims=" << d0 << "x" << d1 << "x" << d2 << " (3D)" << std::endl;
      return false;
    }
  } else {
    return false;
  }

  if (stride < 6 || ((stride - 6) % 3 != 0)) {
    std::cout << "[POSE][yolo26] stride mismatch: stride=" << stride << std::endl;
    return false;
  }
  const int kpts = (stride - 6) / 3;

  const float inW = static_cast<float>(net.width);
  const float inH = static_cast<float>(net.height);
  const float src_w = env_or_default("SQUEAKVIEW_SRC_W", inW);
  const float src_h = env_or_default("SQUEAKVIEW_SRC_H", inH);
  const float gain = std::min(inW / src_w, inH / src_h);
  const float pad_x = 0.5f * (inW - src_w * gain);
  const float pad_y = 0.5f * (inH - src_h * gain);

  dets.clear();
  dets.reserve(num_preds);
  objects.clear();
  objects.reserve(num_preds);

  std::vector<float> row;
  if (channel_major) {
    row.resize(static_cast<size_t>(stride));
  }

  for (int i = 0; i < num_preds; ++i) {
    const float* p = nullptr;
    if (channel_major) {
      for (int c = 0; c < stride; ++c) {
        row[c] = data[c * num_preds + i];
      }
      p = row.data();
    } else {
      p = data + i * stride;
    }

    float x1 = p[0];
    float y1 = p[1];
    float x2 = p[2];
    float y2 = p[3];
    float obj = p[4];
    int cls = static_cast<int>(std::lround(p[5]));
    if (cls < 0) continue;
    if (params.numClassesConfigured > 0 && static_cast<unsigned int>(cls) >= params.numClassesConfigured) {
      continue;
    }
    float thr = class_threshold(params, cls, conf_thr);
    if (obj < thr) continue;

    // Unletterbox xyxy coords from net space back to src space.
    unletterbox(x1, y1, gain, pad_x, pad_y, src_w, src_h);
    unletterbox(x2, y2, gain, pad_x, pad_y, src_w, src_h);
    float bx1 = std::min(x1, x2);
    float by1 = std::min(y1, y2);
    float bx2 = std::max(x1, x2);
    float by2 = std::max(y1, y2);

    PoseDet d{};
    d.cls = cls;
    d.conf = obj;
    d.x1 = bx1;
    d.y1 = by1;
    d.x2 = bx2;
    d.y2 = by2;
    d.kpts.resize(static_cast<size_t>(3 * kpts));
    const float* kp = p + 6;
    for (int k = 0; k < kpts; ++k) {
      float kx = kp[3 * k + 0];
      float ky = kp[3 * k + 1];
      float ks = kp[3 * k + 2];
      unletterbox(kx, ky, gain, pad_x, pad_y, src_w, src_h);
      d.kpts[3 * k + 0] = kx;
      d.kpts[3 * k + 1] = ky;
      d.kpts[3 * k + 2] = ks;
    }
    dets.emplace_back(std::move(d));

    NvDsInferObjectDetectionInfo o{};
    o.classId = cls;
    o.left = bx1;
    o.top = by1;
    o.width = std::max(0.f, bx2 - bx1);
    o.height = std::max(0.f, by2 - by1);
    o.detectionConfidence = obj;
    objects.emplace_back(o);
  }

  // DeepStream 9.1 ServiceMaker consumes output-tensor-meta frame by frame.
  // Do not publish YOLO26 poses through the legacy process-global cache; the
  // parser remains enabled only to preserve nvinfer's detector contract for
  // nvtracker, and its generated objects are filtered before downstream use.
  return true;
}
extern "C" bool NvDsInferParseYolo26Pose(
  const std::vector<NvDsInferLayerInfo>& layers,
  const NvDsInferNetworkInfo& net,
  const NvDsInferParseDetectionParams& params,
  std::vector<NvDsInferObjectDetectionInfo>& objects)
{
  if (layers.empty()) return false;
  const NvDsInferLayerInfo* L = &layers[0];
  for (auto& li : layers) {
    if (li.dataType == NvDsInferDataType::FLOAT) { L = &li; break; }
  }
  std::vector<PoseDet> dets;
  return decode_yolo26_pose(*L, net, params, dets, objects);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Pose);
