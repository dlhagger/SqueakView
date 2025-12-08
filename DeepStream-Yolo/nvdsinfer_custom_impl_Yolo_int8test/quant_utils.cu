#include "quant_utils.h"

namespace {
__global__ void int8ToFloatKernel(const int8_t* input, float* output, float scale, std::size_t count) {
  std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    output[idx] = static_cast<float>(input[idx]) * scale;
  }
}
}

cudaError_t cudaInt8ToFloat(const void* input, float* output, float scale, std::size_t count, cudaStream_t stream) {
  if (scale == 0.0f) {
    scale = 1.0f / 128.0f;
  }
  std::size_t threads = 256;
  std::size_t blocks = (count + threads - 1) / threads;
  int8ToFloatKernel<<<blocks, threads, 0, stream>>>(static_cast<const int8_t*>(input), output, scale, count);
  return cudaGetLastError();
}
