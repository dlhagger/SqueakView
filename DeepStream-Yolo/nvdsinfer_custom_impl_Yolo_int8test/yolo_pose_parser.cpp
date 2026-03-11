// yolo_pose_parser.cpp  (modern YOLOv8/YOLO11 pose, COCO 17 kpts by default)
// Decodes one output tensor shaped [N, 5 + nc + 3*kpts] with [cx,cy,w,h,obj, cls..., kpts(x,y,c)*kpts].
// Exports both NvDsInferParseYoloV8Pose and NvDsInferParseYoloV8PoseBoxes.
// Compile into libnvdsinfer_custom_impl_Yolo.so (Makefile edits below).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <vector>
#include "nvdsinfer_custom_impl.h"

struct PoseDet {
  float x1,y1,x2,y2, conf; int cls;
  std::vector<float> kpts; // size 3*kpts: x,y,score (in input-pixel coords)
};

namespace {

struct PoseCache {
  std::mutex mtx;
  uint64_t seq{0};
  int kpts{0};
  std::vector<float> flat;
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
    std::cout << "[POSE][parser:int8] seq=" << g_pose_cache.seq
              << " dets=" << dets.size()
              << " conf=" << first.conf
              << " kp0=" << kp0 << std::endl;
  } else {
    std::cout << "[POSE][parser:int8] seq=" << g_pose_cache.seq << " dets=0" << std::endl;
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

static bool decode(const NvDsInferLayerInfo& L,
                   const NvDsInferNetworkInfo& net,
                   std::vector<PoseDet>& out,
                   float conf_thr=0.25f, float iou_thr=0.45f)
{
  if (!L.buffer) return false;
  const float* data = static_cast<const float*>(L.buffer);

  int num_preds = 0, dim = 0;
  if (L.inferDims.numDims == 2) { num_preds = L.inferDims.d[0]; dim = L.inferDims.d[1]; }
  else if (L.inferDims.numDims == 3) { num_preds = L.inferDims.d[1]; dim = L.inferDims.d[2]; }
  else return false;

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

  std::vector<PoseDet> dets; dets.reserve(num_preds);
  static bool debug_raw_printed = false;
  static bool debug_det_printed = false;

  for (int i=0;i<num_preds;++i) {
    const float* p = data + i*dim;
    float cx=p[0], cy=p[1], w=p[2], h=p[3], obj=p[4];
    if (!debug_raw_printed) {
      std::cout << "[POSE][parser:int8] raw row0: ";
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
      std::cout << "[POSE][parser:int8] det row center=(" << cx << "," << cy << ") size=(" << w << "," << h
                << ") obj=" << obj << " bestSc=" << bestSc << " conf=" << conf << std::endl;
      for (int k=0;k< std::min(kpts,3); ++k) {
        const float* kp = p + 5 + nc + 3*k;
        std::cout << "   kp" << k << ": [" << kp[0] << ", " << kp[1] << ", " << kp[2] << "]" << std::endl;
      }
      debug_det_printed = true;
    }
    float x1 = cx - 0.5f*w, y1 = cy - 0.5f*h, x2 = cx + 0.5f*w, y2 = cy + 0.5f*h;
    d.x1 = std::min(std::max(x1,0.f), inW-1.f);
    d.y1 = std::min(std::max(y1,0.f), inH-1.f);
    d.x2 = std::min(std::max(x2,0.f), inW-1.f);
    d.y2 = std::min(std::max(y2,0.f), inH-1.f);

    const float* kp = p + 5 + nc;
    d.kpts.resize(3*kpts);
    for (int k=0;k<kpts;++k) {
      float kx=kp[3*k+0], ky=kp[3*k+1], ks=kp[3*k+2];
      kx = std::min(std::max(kx,0.f), inW-1.f);
      ky = std::min(std::max(ky,0.f), inH-1.f);
      d.kpts[3*k+0]=kx; d.kpts[3*k+1]=ky; d.kpts[3*k+2]=ks;
    }
    dets.emplace_back(std::move(d));
  }

  // NMS
  std::sort(dets.begin(), dets.end(), [](const PoseDet&a,const PoseDet&b){return a.conf>b.conf;});
  std::vector<char> rem(dets.size(),0); std::vector<PoseDet> keep; keep.reserve(dets.size());
  for (size_t i=0;i<dets.size();++i) {
    if (rem[i]) continue; keep.push_back(dets[i]);
    for (size_t j=i+1;j<dets.size();++j) if (!rem[j] && iou_xyxy(dets[i],dets[j])>iou_thr) rem[j]=1;
  }
  update_pose_cache(keep, kpts);
  out.swap(keep);
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
