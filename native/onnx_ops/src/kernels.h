#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
void RowMax(cudaStream_t stream, const float* input, float* values, int64_t* indices,
            int64_t rows, int64_t width, bool largest);
