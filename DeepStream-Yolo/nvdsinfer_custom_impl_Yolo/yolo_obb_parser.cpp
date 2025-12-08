// yolo_obb_parser.cpp
// DeepStream nvinfer postprocessor for Ultralytics YOLO11-OBB / YOLOv8-OBB
// Decodes one output tensor shaped [N, D] where D = 5 + 1 + nc  (cx,cy,w,h,theta, obj, class_scores...)
// Some exports place 'obj' before theta: [cx,cy,w,h,obj,theta, class_scores...].
// This parser supports BOTH layouts via a small runtime check.
//
// It returns axis-aligned boxes to DeepStream (so you get immediate OSD),
// and (optionally) attaches the 5-tuple OBB (cx,cy,w,h,theta) as user meta for a
// later overlay step if you want true rotated drawing.
//
// Exports two common symbol names:
//   NvDsInferParseYoloV8OBB
//   NvDsInferParseYoloOBB
//
// Build: the Makefile in this repo auto-picks up *.cpp, so just `make -j` in the folder.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <cstring>
#include <string>

#include "nvdsinfer_custom_impl.h"
#include "nvdsinfer.h"

static inline float clampf(float v, float lo, float hi) {
    return std::min(std::max(v, lo), hi);
}

struct OBBDet {
    float cx, cy, w, h, theta; // theta in radians
    float conf;
    int   cls;
};

// IoU on AABB for NMS (fast + compatible with DS OSD)
static inline float iou_aabb(const OBBDet& a, const OBBDet& b) {
    float ax1 = a.cx - a.w*0.5f, ay1 = a.cy - a.h*0.5f;
    float ax2 = a.cx + a.w*0.5f, ay2 = a.cy + a.h*0.5f;
    float bx1 = b.cx - b.w*0.5f, by1 = b.cy - b.h*0.5f;
    float bx2 = b.cx + b.w*0.5f, by2 = b.cy + b.h*0.5f;
    float xx1 = std::max(ax1, bx1), yy1 = std::max(ay1, by1);
    float xx2 = std::min(ax2, bx2), yy2 = std::min(ay2, by2);
    float w = std::max(0.0f, xx2 - xx1), h = std::max(0.0f, yy2 - yy1);
    float inter = w*h;
    float areaA = std::max(0.0f, ax2-ax1) * std::max(0.0f, ay2-ay1);
    float areaB = std::max(0.0f, bx2-bx1) * std::max(0.0f, by2-by1);
    float uni = areaA + areaB - inter + 1e-6f;
    return inter / uni;
}

// Try to decode one vector [D] into OBBDet.
// Supports TWO common layouts:
//  L0: [cx,cy,w,h,theta, obj, cls...]   (Ultralytics typical)
//  L1: [cx,cy,w,h, obj, theta, cls...]  (some exports)
// Returns false if conf below threshold or dims invalid.
static bool decode_one(const float* p, int D, int nc, float inW, float inH,
                       float conf_thr, OBBDet& out) {
    if (D < 6 + nc) return false;
    float cx = p[0], cy = p[1], w = p[2], h = p[3];
    // Decide where obj/theta live:
    // Heuristic: theta should be in ~[-pi, pi], obj in [0,1].
    float cand_theta0 = p[4];
    float cand_obj0   = p[5];
    float cand_obj1   = p[4];
    float cand_theta1 = p[5];

    bool layout0 = (std::fabs(cand_theta0) <= 3.5f) && (cand_obj0 >= 0.f && cand_obj0 <= 1.0001f);
    bool layout1 = (std::fabs(cand_theta1) <= 3.5f) && (cand_obj1 >= 0.f && cand_obj1 <= 1.0001f);

    float theta = 0.f, obj = 0.f;
    int cls_offset = 0;
    if (layout0 && !layout1) {
        theta = cand_theta0; obj = cand_obj0; cls_offset = 6;
    } else if (layout1 && !layout0) {
        obj = cand_obj1; theta = cand_theta1; cls_offset = 6;
    } else {
        // fallback: assume Ultralytics (theta at [4], obj at [5])
        theta = cand_theta0; obj = cand_obj0; cls_offset = 6;
    }

    if (obj < conf_thr) return false;

    // class picking
    int bestId = 0; float bestSc = 1.f;
    if (nc > 1) {
        bestSc = 0.f;
        for (int c=0; c<nc; ++c) {
            float sc = p[cls_offset + c];
            if (sc > bestSc) { bestSc = sc; bestId = c; }
        }
    }
    float conf = obj * bestSc;
    if (conf < conf_thr) return false;

    // clamp to input dims (engine input coordinates)
    OBBDet d{};
    d.cx = clampf(cx, 0.f, inW - 1.f);
    d.cy = clampf(cy, 0.f, inH - 1.f);
    d.w  = std::max(0.f, std::min(std::fabs(w), inW));
    d.h  = std::max(0.f, std::min(std::fabs(h), inH));
    d.theta = theta;    // radians (expected)
    d.conf  = conf;
    d.cls   = bestId;
    out = d;
    return true;
}

// dim = 5 (cx,cy,w,h,theta) + 1 (obj) + nc
static bool decode_all(const NvDsInferLayerInfo& L,
                       const NvDsInferNetworkInfo& net,
                       std::vector<OBBDet>& out,
                       float conf_thr = 0.25f, float iou_thr = 0.45f) {
    if (!L.buffer) return false;
    const float* data = static_cast<const float*>(L.buffer);

    int N=0, D=0;
    if (L.inferDims.numDims == 2) { N = L.inferDims.d[0]; D = L.inferDims.d[1]; }
    else if (L.inferDims.numDims == 3) { N = L.inferDims.d[1]; D = L.inferDims.d[2]; }
    else return false;

    // infer nc from D
    int nc = D - 6;  // assume [5 obb + 1 obj] + nc
    if (nc < 1) nc = 1;  // guard

    const float inW = static_cast<float>(net.width);
    const float inH = static_cast<float>(net.height);

    std::vector<OBBDet> dets; dets.reserve(N);
    for (int i=0; i<N; ++i) {
        const float* p = data + i*D;
        OBBDet d;
        if (decode_one(p, D, nc, inW, inH, conf_thr, d)) {
            dets.emplace_back(d);
        }
    }

    // NMS on AABB
    std::sort(dets.begin(), dets.end(), [](const OBBDet&a,const OBBDet&b){return a.conf>b.conf;});
    std::vector<char> rem(dets.size(),0); std::vector<OBBDet> keep; keep.reserve(dets.size());
    for (size_t i=0;i<dets.size();++i) {
        if (rem[i]) continue; keep.push_back(dets[i]);
        for (size_t j=i+1;j<dets.size();++j) if (!rem[j] && iou_aabb(dets[i], dets[j])>iou_thr) rem[j]=1;
    }
    out.swap(keep);
    return true;
}

static bool parse_obb_internal(const std::vector<NvDsInferLayerInfo>& layers,
                               const NvDsInferNetworkInfo& net,
                               const NvDsInferParseDetectionParams& /*params*/,
                               std::vector<NvDsInferObjectDetectionInfo>& objects)
{
    if (layers.empty()) return false;
    const NvDsInferLayerInfo* L = &layers[0];
    for (auto& li: layers) if (li.dataType == NvDsInferDataType::FLOAT) { L=&li; break; }

    std::vector<OBBDet> dets;
    if (!decode_all(*L, net, dets)) return false;

    objects.clear(); objects.reserve(dets.size());
    for (const auto& d : dets) {
        NvDsInferObjectDetectionInfo o{};
        o.classId = d.cls;
        o.detectionConfidence = d.conf;
        // Convert to AABB for DS box drawing now
        float x1 = d.cx - d.w*0.5f, y1 = d.cy - d.h*0.5f;
        float w  = std::max(0.f, d.w);
        float h  = std::max(0.f, d.h);
        o.left = x1; o.top = y1; o.width = w; o.height = h;
        objects.emplace_back(o);
        // NOTE: If you want rotated drawing, we can add user meta here
        // with (cx,cy,w,h,theta) and consume it in a small OSD adapter.
    }
    return true;
}

extern "C" bool NvDsInferParseYoloV8OBB(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& net,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects)
{
    return parse_obb_internal(layers, net, params, objects);
}

extern "C" bool NvDsInferParseYoloOBB(
    const std::vector<NvDsInferLayerInfo>& layers,
    const NvDsInferNetworkInfo& net,
    const NvDsInferParseDetectionParams& params,
    std::vector<NvDsInferObjectDetectionInfo>& objects)
{
    return parse_obb_internal(layers, net, params, objects);
}

