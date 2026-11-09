// Compare the exported ABI against an immutable baseline on identical tensors.
#include "nvdsinfer_custom_impl.h"
#include <dlfcn.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using Parser = bool (*)(const std::vector<NvDsInferLayerInfo>&, const NvDsInferNetworkInfo&,
                       const NvDsInferParseDetectionParams&, std::vector<NvDsInferObjectDetectionInfo>&);
struct Library {
  void* handle;
  Parser parse;
  explicit Library(const char* path) : handle(dlopen(path, RTLD_NOW | RTLD_LOCAL)) {
    if (!handle) throw std::runtime_error(dlerror());
    parse = reinterpret_cast<Parser>(dlsym(handle, "NvDsInferParseYolo26Pose"));
    if (!parse) throw std::runtime_error("missing parser entry point");
  }
  ~Library() { dlclose(handle); }
};

static size_t compare(Library& old, Library& candidate, std::vector<float>& data,
                     int rows, int columns, bool channels_first, bool batch = false, bool require_success = false) {
  NvDsInferLayerInfo layer{};
  layer.dataType = NvDsInferDataType::FLOAT;
  layer.buffer = data.data();
  layer.inferDims.numDims = batch ? 3 : 2;
  int offset = batch ? 1 : 0;
  if (batch) layer.inferDims.d[0] = 1;
  layer.inferDims.d[offset] = channels_first ? columns : rows;
  layer.inferDims.d[offset + 1] = channels_first ? rows : columns;
  layer.inferDims.numElements = data.size();
  NvDsInferNetworkInfo net{};
  net.width = net.height = 640;
  net.channels = 3;
  NvDsInferParseDetectionParams params{};
  params.numClassesConfigured = 2;
  params.perClassPreclusterThreshold = {.25f, .45f};
  std::vector<NvDsInferObjectDetectionInfo> a, b;
  bool ok_a = old.parse({layer}, net, params, a);
  bool ok_b = candidate.parse({layer}, net, params, b);
  if (require_success && (!ok_a || !ok_b)) throw std::runtime_error("real tensor rejected by parser");
  if (ok_a != ok_b || a.size() != b.size()) throw std::runtime_error("status/count mismatch");
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].classId != b[i].classId || a[i].left != b[i].left || a[i].top != b[i].top ||
        a[i].width != b[i].width || a[i].height != b[i].height ||
        a[i].detectionConfidence != b[i].detectionConfidence)
      throw std::runtime_error("detection mismatch");
  }
  return a.size();
}

int main(int argc, char** argv) try {
  if (argc != 3 && argc != 6) throw std::runtime_error("usage: compare_pose_parsers BASELINE CANDIDATE [FLOAT32_FILE ROWS COLUMNS]");
  Library old(argv[1]), candidate(argv[2]);
  if (argc == 6) {
    int rows = std::stoi(argv[4]), columns = std::stoi(argv[5]);
    if (rows < 1 || rows > 10000 || columns < 6 || columns > 4096) throw std::runtime_error("invalid tensor dimensions");
    std::vector<float> data(static_cast<size_t>(rows) * columns);
    std::ifstream file(argv[3], std::ios::binary);
    if (!file.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float)) || file.peek() != EOF)
      throw std::runtime_error("tensor file size mismatch");
    std::cout << "PASS real tensor: " << compare(old, candidate, data, rows, columns, false, false, true) << " detections identical\n";
    return 0;
  }
  std::mt19937 rng(26);
  std::uniform_real_distribution<float> unit(0, 1), coord(-100, 750);
  size_t cases = 0;
  for (const char* width : {"640", "1440", "1920"}) {
    setenv("SQUEAKVIEW_SRC_W", width, 1);
    setenv("SQUEAKVIEW_SRC_H", "1080", 1);
    for (int keypoints : {1, 17, 19}) {
      int rows = 300, columns = 6 + 3 * keypoints;
      for (int iteration = 0; iteration < 20; ++iteration) {
        std::vector<float> data(rows * columns);
        for (int row = 0; row < rows; ++row) {
          for (int c = 0; c < columns; ++c) data[row * columns + c] = coord(rng);
          data[row * columns + 4] = unit(rng);
          data[row * columns + 5] = row % 3; // includes unconfigured classes
          for (int k = 0; k < keypoints; ++k) data[row * columns + 8 + 3 * k] = unit(rng);
        }
        for (bool batch : {false, true}) {
          compare(old, candidate, data, rows, columns, false, batch); ++cases;
          std::vector<float> transposed(data.size());
          for (int r = 0; r < rows; ++r) for (int c = 0; c < columns; ++c)
            transposed[c * rows + r] = data[r * columns + c];
          compare(old, candidate, transposed, rows, columns, true, batch); ++cases;
        }
        if (iteration == 0) {
          data[0] = std::numeric_limits<float>::quiet_NaN();
          compare(old, candidate, data, rows, columns, false); ++cases;
        }
      }
    }
  }
  unsetenv("SQUEAKVIEW_SRC_W"); unsetenv("SQUEAKVIEW_SRC_H");
  std::cout << "PASS: " << cases << " tensor cases matched exactly\n";
  return 0;
} catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
