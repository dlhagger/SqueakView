// CPU-only boundary tests against the production parser. No CUDA execution.
#include "../yolo_pose_parser.cpp"
#include <cassert>

int main() {
  unsetenv("SQUEAKVIEW_SRC_W");
  unsetenv("SQUEAKVIEW_SRC_H");
  NvDsInferNetworkInfo net{};
  net.width = net.height = 640;
  NvDsInferParseDetectionParams params{};
  params.numClassesConfigured = 1;
  params.perClassPreclusterThreshold = {0.25f};
  std::vector<float> data = {10, 20, 100, 200, .9f, 0, 50, 60, .8f};
  NvDsInferLayerInfo layer{};
  layer.buffer = data.data();
  layer.dataType = NvDsInferDataType::FLOAT;
  layer.inferDims.numDims = 2;
  layer.inferDims.d[0] = 1;
  layer.inferDims.d[1] = 9;
  layer.inferDims.numElements = 9;
  std::vector<NvDsInferObjectDetectionInfo> objects;
  auto parse = [&]() { return NvDsInferParseYolo26Pose({layer}, net, params, objects); };
  assert(parse() && objects.size() == 1 && objects[0].width == 90);
  for (size_t index = 0; index < data.size(); ++index) {
    float saved = data[index];
    for (float invalid : {NAN, INFINITY, -INFINITY}) {
      data[index] = invalid;
      assert(!parse() && objects.empty());
    }
    data[index] = saved;
  }
  for (size_t index : {size_t(4), size_t(8)}) {
    float saved = data[index];
    for (float invalid : {-0.1f, 1.1f}) {
      data[index] = invalid;
      assert(!parse() && objects.empty());
    }
    data[index] = saved;
  }
  for (float invalid : {-1.f, .5f, 3e9f}) {
    data[5] = invalid;
    assert(!parse() && objects.empty());
  }
  data[5] = 0;
  data[4] = .1f;
  assert(parse() && objects.empty());
  data[4] = .9f;
  params.perClassPreclusterThreshold[0] = NAN;
  assert(!parse() && objects.empty());
  params.perClassPreclusterThreshold[0] = .25f;
  layer.dataType = NvDsInferDataType::HALF;
  assert(!parse() && objects.empty());
  layer.dataType = NvDsInferDataType::FLOAT;
  layer.inferDims.numElements = 8;
  assert(!parse());
  layer.inferDims.numElements = 9;
  layer.inferDims.d[0] = -1;
  assert(!parse());
  layer.inferDims.d[0] = 1;
  net.width = 0;
  assert(!parse());
  net.width = 640;
  for (const char* invalid : {"0", "-10", "nan", "inf"}) {
    setenv("SQUEAKVIEW_SRC_W", invalid, 1);
    assert(!parse());
  }
  setenv("SQUEAKVIEW_SRC_W", "1280", 1);
  setenv("SQUEAKVIEW_SRC_H", "640", 1);
  assert(parse() && objects.size() == 1);
  assert(objects[0].left == 20 && objects[0].top == 0);
  unsetenv("SQUEAKVIEW_SRC_W");
  unsetenv("SQUEAKVIEW_SRC_H");
  // Channels-first one-frame input and explicit batch rejection.
  layer.inferDims.numDims = 3;
  layer.inferDims.d[0] = 1;
  layer.inferDims.d[1] = 9;
  layer.inferDims.d[2] = 1;
  assert(parse() && objects.size() == 1);
  layer.inferDims.d[0] = 2;
  assert(!parse());
  // Invalid second row must remove an already decoded first detection.
  data.insert(data.end(), {10, 20, 100, 200, NAN, 0, 50, 60, .8f});
  layer.buffer = data.data();
  layer.inferDims.numDims = 2;
  layer.inferDims.d[0] = 2;
  layer.inferDims.d[1] = 9;
  layer.inferDims.numElements = 18;
  assert(!parse() && objects.empty());
  return 0;
}
