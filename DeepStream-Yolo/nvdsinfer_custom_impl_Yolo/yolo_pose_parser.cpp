// yolo_pose_parser.cpp  (modern YOLOv8/YOLO11 pose, COCO 17 kpts by default)
// Decodes one output tensor shaped [N, 5 + nc + 3*kpts] with [cx,cy,w,h,obj, cls..., kpts(x,y,c)*kpts].
// Exports both NvDsInferParseYoloV8Pose and NvDsInferParseYoloV8PoseBoxes.
// Compile into libnvdsinfer_custom_impl_Yolo.so (Makefile edits below).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <vector>
#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer.h"

struct PoseDet {
  float x1,y1,x2,y2, conf; int cls;
  std::vector<float> kpts; // size 3*kpts: x,y,score (in input-pixel coords)
};

namespace {

struct PoseCache {
  std::mutex mtx;
  uint64_t seq{0};
  int kpts{0};
  std::vector<float> flat; // [x1,y1,x2,y2,conf, kpts...]
} g_pose_cache;

constexpr int kBaseValuesPerDet = 5;

void update_pose_cache(const std::vector<PoseDet>& dets, int kpts) {
  std::lock_guard<std::mutex> lock(g_pose_cache.mtx);
  g_pose_cache.seq++;
  g_pose_cache.kpts = kpts;
  const int stride = kBaseValuesPerDet + 3 * std::max(0, kpts);
  g_pose_cache.flat.clear();
  g_pose_cache.flat.reserve(static_cast<size_t>(dets.size()) * stride);
  for (const auto& d : dets) {
    g_pose_cache.flat.push_back(d.x1);
    g_pose_cache.flat.push_back(d.y1);
    g_pose_cache.flat.push_back(d.x2);
    g_pose_cache.flat.push_back(d.y2);
    g_pose_cache.flat.push_back(d.conf);
    g_pose_cache.flat.insert(g_pose_cache.flat.end(), d.kpts.begin(), d.kpts.end());
  }
  std::cout << std::fixed << std::setprecision(4);
  if (!dets.empty()) {
    const auto& first = dets.front();
    float kp0 = first.kpts.empty() ? 0.f : first.kpts[0];
    std::cout << "[POSE][parser] seq=" << g_pose_cache.seq
              << " dets=" << dets.size()
              << " conf=" << first.conf
              << " kp0=" << kp0 << std::endl;
  } else {
    std::cout << "[POSE][parser] seq=" << g_pose_cache.seq << " dets=0" << std::endl;
  }
}

} // namespace

extern "C" uint64_t NvDsInferGetPoseCache(float** data, int* count, int* kpts) {
  std::lock_guard<std::mutex> lock(g_pose_cache.mtx);
  if (data) {
    *data = g_pose_cache.flat.empty() ? nullptr : g_pose_cache.flat.data();
  }
  if (count) {
    *count = static_cast<int>(g_pose_cache.flat.size());
  }
  if (kpts) {
    *kpts = g_pose_cache.kpts;
  }
  return g_pose_cache.seq;
}

static inline float iou_xyxy(const PoseDet& a, const PoseDet& b) {
  float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
  float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
  float w = std::max(0.f, xx2-xx1), h = std::max(0.f, yy2-yy1);
  float inter = w*h;
  float areaA = std::max(0.f, a.x2-a.x1)*std::max(0.f, a.y2-a.y1);
  float areaB = std::max(0.f, b.x2-b.x1)*std::max(0.f, b.y2-b.y1);
  return inter / (areaA + areaB - inter + 1e-6f);
}

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

static bool decode(const NvDsInferLayerInfo& L,
                   const NvDsInferNetworkInfo& net,
                   std::vector<PoseDet>& out,
                   float conf_thr=0.25f, float iou_thr=0.45f)
{
  if (!L.buffer) return false;
  const float* data = static_cast<const float*>(L.buffer);

  int num_preds = 0, dim = 0;
  bool channel_major = false;
  int stride = 0;

  if (L.inferDims.numDims == 2) {
    int a = L.inferDims.d[0];
    int b = L.inferDims.d[1];
    // Common exported shape is [C, N] with C=5+nc+3*kpts, N=anchors.
    if (a < b) { dim = a; num_preds = b; channel_major = true; stride = num_preds; }
    else { num_preds = a; dim = b; stride = dim; }
    std::cout << "[POSE][parser] dims=" << a << "x" << b
              << " (2D) -> num_preds=" << num_preds << " dim=" << dim
              << " channel_major=" << (channel_major ? 1 : 0) << std::endl;
  } else if (L.inferDims.numDims == 3) {
    // Assume [B, C, N] as exported by Ultralytics pose: C = 5 + nc + 3*kpts, N = anchors.
    dim = L.inferDims.d[1];
    num_preds = L.inferDims.d[2];
    channel_major = true;              // data laid out as channel-major
    stride = num_preds;                // step between channel values
    std::cout << "[POSE][parser] dims=" << L.inferDims.d[0] << "x" << dim << "x" << num_preds
              << " (channel-major) -> num_preds=" << num_preds << " dim=" << dim << std::endl;
  } else {
    return false;
  }

  if (dim < 5 + 3) return false; // sanity: needs at least obj + 1 kpt triplet

  // Auto-infer (nc, kpts) given dim = 5 + nc + 3*kpts
  int max_kpts = 50; // guard
  int nc = 1, kpts = 17;
  bool inferred=false;
  int fallback_nc = -1, fallback_k = 0;
  for (int guess_k=1; guess_k<=max_kpts; ++guess_k) {
    int rem = dim - 5 - 3*guess_k;
    if (rem < 0) continue;
    if (rem == 0) {
      nc = 0;
      kpts = guess_k;
      inferred = true;
      break;
    }
    if (fallback_nc < 0) {
      fallback_nc = rem;
      fallback_k = guess_k;
    }
  }
  if (!inferred) {
    if (fallback_nc < 0) return false;
    nc = fallback_nc;
    kpts = fallback_k;
  }

  const float inW = static_cast<float>(net.width);
  const float inH = static_cast<float>(net.height);
  // Allow overriding source frame size via env so we can unletterbox correctly.
  const float src_w = env_or_default("SQUEAKVIEW_SRC_W", inW);
  const float src_h = env_or_default("SQUEAKVIEW_SRC_H", inH);
  const float gain  = std::min(inW / src_w, inH / src_h);
  const float pad_x = 0.5f * (inW - src_w * gain);
  const float pad_y = 0.5f * (inH - src_h * gain);
  static bool dbg_geom = false;
  if (!dbg_geom) {
    std::cout << "[POSE][parser] geom src=(" << src_w << "x" << src_h << ") net=("
              << inW << "x" << inH << ") gain=" << gain
              << " pad=(" << pad_x << "," << pad_y << ")\n";
    dbg_geom = true;
  }

  std::vector<PoseDet> dets; dets.reserve(num_preds);
  static bool debug_raw_printed = false;
  static bool debug_det_printed = false;

  std::vector<float> row(dim);

  for (int i=0;i<num_preds;++i) {
    const float* p = nullptr;
    if (channel_major) {
      for (int c=0;c<dim;++c) {
        row[c] = data[c*stride + i];
      }
      p = row.data();
    } else {
      p = data + i*dim;
    }

    float cx=p[0], cy=p[1], w=p[2], h=p[3], obj=p[4];
    if (!debug_raw_printed) {
      std::cout << "[POSE][parser] raw row0: ";
      for (int t=0; t<std::min(dim, 32); ++t) {
        std::cout << p[t] << (t+1 < std::min(dim, 32) ? ", " : "");
      }
      std::cout << std::endl;
      debug_raw_printed = true;
    }
    if (obj < conf_thr) continue;

    // class score
    int bestId = 0; float bestSc = 1.f;
    if (nc>1) {
      bestSc = 0.f;
      for (int c=0;c<nc;++c) { float sc=p[5+c]; if (sc>bestSc){bestSc=sc; bestId=c;} }
    }
    float conf = obj * bestSc;
    if (conf < conf_thr) continue;

    PoseDet d{};
    d.cls = bestId; d.conf = conf;
    if (!debug_det_printed) {
      std::cout << "[POSE][parser] det row center=(" << cx << "," << cy << ") size=(" << w << "," << h
                << ") obj=" << obj << " bestSc=" << bestSc << " conf=" << conf << std::endl;
      for (int k=0;k< std::min(kpts,3); ++k) {
        const float* kp = p + 5 + nc + 3*k;
        std::cout << "   kp" << k << ": [" << kp[0] << ", " << kp[1] << ", " << kp[2] << "]" << std::endl;
      }
      debug_det_printed = true;
    }
    // Some exports output xyxy instead of cxcywh. Heuristically detect if width/height look like coords.
    bool xyxy = (w > inW) || (h > inH) || (cx > inW) || (cy > inH);
    float x1, y1, x2, y2;
    if (xyxy) {
      x1 = cx; y1 = cy; x2 = w; y2 = h;
    } else {
      x1 = cx - 0.5f*w; y1 = cy - 0.5f*h; x2 = cx + 0.5f*w; y2 = cy + 0.5f*h;
    }
    d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
    if (!debug_det_printed) {
      std::cout << "[POSE][parser] pre-unletterbox box=(" << x1 << "," << y1 << ")-("
                << x2 << "," << y2 << ")\n";
    }
    unletterbox(d.x1, d.y1, gain, pad_x, pad_y, src_w, src_h);
    unletterbox(d.x2, d.y2, gain, pad_x, pad_y, src_w, src_h);

    const float* kp = p + 5 + nc;
    d.kpts.resize(3*kpts);
    for (int k=0;k<kpts;++k) {
      float kx=kp[3*k+0], ky=kp[3*k+1], ks=kp[3*k+2];
      unletterbox(kx, ky, gain, pad_x, pad_y, src_w, src_h);
      d.kpts[3*k+0]=kx; d.kpts[3*k+1]=ky; d.kpts[3*k+2]=ks;
    }
    dets.emplace_back(std::move(d));
  }

  // NMS
  std::sort(dets.begin(), dets.end(), [](const PoseDet&a,const PoseDet&b){return a.conf>b.conf;});
  std::vector<char> rem(dets.size(),0); std::vector<PoseDet> keep; keep.reserve(dets.size());
  for (size_t i=0;i<dets.size();++i) {
    if (rem[i]) continue;
    keep.push_back(dets[i]);
    for (size_t j=i+1;j<dets.size();++j) {
      if (!rem[j] && iou_xyxy(dets[i],dets[j])>iou_thr) rem[j]=1;
    }
  }
  std::cout << "[POSE][parser] preds=" << num_preds << " dim=" << dim
            << " dets_before_nms=" << dets.size() << " dets_after_nms=" << keep.size()
            << " channel_major=" << (channel_major ? 1 : 0) << std::endl;
  update_pose_cache(keep, kpts);
  out.swap(keep);
  return true;
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
                               std::vector<NvDsInferInstanceMaskInfo>& objects,
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

    NvDsInferInstanceMaskInfo o{};
    o.classId = static_cast<unsigned int>(cls);
    o.left = bx1;
    o.top = by1;
    o.width = std::max(0.f, bx2 - bx1);
    o.height = std::max(0.f, by2 - by1);
    o.detectionConfidence = obj;
    o.mask = nullptr;
    o.mask_width = 0;
    o.mask_height = 0;
    o.mask_size = 0;
    objects.emplace_back(o);
  }

  update_pose_cache(dets, kpts);
  return true;
}

static bool parse_pose_internal(const std::vector<NvDsInferLayerInfo>& layers,
                                const NvDsInferNetworkInfo& net,
                                const NvDsInferParseDetectionParams&,
                                std::vector<NvDsInferObjectDetectionInfo>& objects)
{
  if (layers.empty()) return false;
  // pick first FP32 layer or fallback
  const NvDsInferLayerInfo* L=&layers[0];
  for (auto& li: layers) if (li.dataType==NvDsInferDataType::FLOAT) { L=&li; break; }

  std::vector<PoseDet> dets;
  if (!decode(*L, net, dets)) return false;

  objects.clear(); objects.reserve(dets.size());
  for (auto& d: dets) {
    NvDsInferObjectDetectionInfo o{};
    o.classId = d.cls; o.detectionConfidence = d.conf;
    o.left = d.x1; o.top = d.y1; o.width = std::max(0.f, d.x2-d.x1); o.height = std::max(0.f, d.y2-d.y1);
    objects.emplace_back(o);
  }
  std::cout << "[POSE][parser] objects_emitted=" << objects.size() << std::endl;
  return true;
}

extern "C" bool NvDsInferParseYoloV8Pose(
  const std::vector<NvDsInferLayerInfo>& layers,
  const NvDsInferNetworkInfo& net,
  const NvDsInferParseDetectionParams& params,
  std::vector<NvDsInferObjectDetectionInfo>& objects)
{
  return parse_pose_internal(layers, net, params, objects);
}

extern "C" bool NvDsInferParseYoloV8PoseBoxes(
  const std::vector<NvDsInferLayerInfo>& layers,
  const NvDsInferNetworkInfo& net,
  const NvDsInferParseDetectionParams& params,
  std::vector<NvDsInferObjectDetectionInfo>& objects)
{
  return parse_pose_internal(layers, net, params, objects);
}

extern "C" bool NvDsInferParseYolo26Pose(
  const std::vector<NvDsInferLayerInfo>& layers,
  const NvDsInferNetworkInfo& net,
  const NvDsInferParseDetectionParams& params,
  std::vector<NvDsInferInstanceMaskInfo>& objects)
{
  if (layers.empty()) return false;
  const NvDsInferLayerInfo* L = &layers[0];
  for (auto& li : layers) {
    if (li.dataType == NvDsInferDataType::FLOAT) { L = &li; break; }
  }
  std::vector<PoseDet> dets;
  return decode_yolo26_pose(*L, net, params, dets, objects);
}

CHECK_CUSTOM_INSTANCE_MASK_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26Pose);
