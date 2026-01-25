#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

cudaError_t cudaInt8ToFloat(const void* input, float* output, float scale, std::size_t count, cudaStream_t stream);
