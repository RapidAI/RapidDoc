#include "kernels.h"
#include <cub/block/block_reduce.cuh>
#include <algorithm>
#include <cmath>
struct Pair { float value; int64_t index; };
struct Compare {
  bool largest;
  __device__ Pair operator()(Pair a, Pair b) const {
    if (a.index < 0) return b;
    if (b.index < 0) return a;
    const bool an = isnan(a.value), bn = isnan(b.value);
    if (an != bn) return an ? a : b;
    if (!an) {
      if (a.value > b.value) return largest ? a : b;
      if (a.value < b.value) return largest ? b : a;
    }
    return a.index <= b.index ? a : b;
  }
};
__global__ void RowMaxKernel(const float* x, float* values, int64_t* indices,
                             int64_t rows, int64_t width, bool largest) {
  using Reduction = cub::BlockReduce<Pair, 256>;
  __shared__ Reduction::TempStorage storage;
  Compare compare{largest};
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    Pair best{0.f, -1};
    for (int64_t col = threadIdx.x; col < width; col += 256)
      best = compare(best, Pair{x[row * width + col], col});
    Pair out = Reduction(storage).Reduce(best, compare);
    if (threadIdx.x == 0) { values[row] = out.value; indices[row] = out.index; }
    __syncthreads();
  }
}
void RowMax(cudaStream_t stream, const float* x, float* v, int64_t* i,
            int64_t rows, int64_t width, bool largest) {
  if (rows) RowMaxKernel<<<static_cast<unsigned>(std::min<int64_t>(rows, 65535)), 256, 0, stream>>>(x,v,i,rows,width,largest);
}
