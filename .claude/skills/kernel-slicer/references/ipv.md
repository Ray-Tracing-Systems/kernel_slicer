# IPV pattern (Image Processing Vectorization)

IPV is a standard "parallel for". It is the default choice.

## Anatomy of an IPV kernel

```cpp
void MyClass::kernel2D_Name(int w, int h, const float4* a_in, float4* a_out)
{
  // (A) prologue: executes ONCE before the grid (initialize reduction members, reset counters)
  m_maxValue = 0.0f;

  for(int y = 0; y < h; y++)          // thread loop #1 (outer, Y)
  {
    for(int x = 0; x < w; x++)        // thread loop #2 (inner, X), perfectly nested
    {
      // (B) body: executes once PER THREAD (x,y)
      float4 c = a_in[y*w + x];
      a_out[y*w + x] = c*0.5f;
      m_maxValue = std::max(m_maxValue, c.x);   // reduction into a member
    }
  }

  // (C) epilogue: executes ONCE after the grid (finalize reduction, write single values)
}
```

Rules:
- For 2D kernels write `y` outer and `x` inner, and translate with `-reorderLoops YX` so that `x`
  becomes the fastest GPU thread index (coalesced memory access; 2–5× faster for memory-bound
  kernels). Without the option the outer loop variable is the fast index.
- The prefix fixes the grid dimension: `kernel1D_` has 1 thread loop, `kernel2D_` has 2 nested
  loops, `kernel3D_` has 3. Loop variable names are free (`i`, `x`, `y`, `tidX`...).
- Loops must be perfectly nested: no statements between the outer `for` and the inner `for`.
- Exactly one thread-loop nest per kernel. Inner loops *inside the body* (filter taps, 4x4
  downsample, iterating over a small member array) are fine; they run inside each thread.
- Loop form: `for(int i = START; i < BOUND; i++)`. `START` may be non-zero
  (`a_size/2`, `1`). `BOUND` must be a **kernel argument** or a **class member**, or
  `m_vec.size()` / `m_vec.capacity()` of a member vector (the translator then generates an
  indirect dispatch). Compute `w*h` in the control function and pass it as an argument:
  `i < w*h` is translated silently into shaders that do not compile.
- Use `<`, not `<=`, for the bound (the translator warns "possible end-of-loop bug" otherwise).
- Kernel pointer arguments get their buffers from the control function (its arguments or
  `m_member.data()`); kernel scalar arguments come from the control function too.
- Kernels read class members directly; they do not need them as arguments.

### Reductions

A scalar or `float4` class member updated in the body with one of these forms becomes a GPU
parallel reduction (sum, min, max), with the prologue as initialization:

```cpp
m_summ += v;          m_summ -= v;       m_count++;
m_min = std::min(m_min, v);    m_boxMin = min(m_boxMin, p);   // LiteMath min/max on float4 too
m_max = std::max(m_max, std::max(a, b));
```

The general form is `a = f(a, b)` where `f` has two arguments and one of them is `a` itself.
Write-only flags such as `if(v > 1000.0) m_flag = 1;` are also accepted.

For many accumulators indexed at run time use atomics on an output pointer or member vector:

```cpp
using LiteMath::InterlockedAdd;
InterlockedAdd(a_out[i % a_size], int(i));   // int, uint and float
```

or the `ReduceAdd` helpers on a member vector (see `apps/tests/045_reduce_add`):

```cpp
// control function
ReduceAddInit(m_accum, m_accum.size());
kernel1D_Accumulate(in_data, n, a_out);
ReduceAddComplete(m_accum);
// kernel body
ReduceAdd<float, uint32_t>(m_accum, binIndex, value);
```

### Appending to a member vector

```cpp
// host, before CommitDeviceData():  m_found.reserve(maxCount);
void RedPixels::kernel1D_Find(const uint32_t* a_data, size_t a_size)
{
  m_found.resize(0);                       // prologue: reset size
  for(uint32_t i = 0; i < a_size; i++)
    if(IsRed(a_data[i]))
      m_found.push_back(PixelInfo{a_data[i], i});
}

void RedPixels::kernel1D_Paint(uint32_t* a_data)   // grid size = number of found elements
{
  for(uint32_t k = 0; k < m_found.size(); k++)
    a_data[m_found[k].index] = 0x0000FFFF;
}
```

Order of appended elements on the GPU is not deterministic.

## Complete example: Reinhard tone mapping (two kernels)

`reinhard.h`
```cpp
#pragma once
#include <vector>
#include <cstdint>
#include <chrono>
#include "LiteMath.h"
using LiteMath::float4;

class ReinhardTM
{
public:
  ReinhardTM() {}

  virtual void Run(int w, int h, const float* inData [[size("w*h*4")]], uint32_t* outData [[size("w*h")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; }

  float getWhitePoint() const { return whitePoint; }

protected:
  virtual void kernel1D_findMax(const float* inData, int size);
  virtual void kernel2D_process(int w, int h, const float* inData, uint32_t* outData);

  float whitePoint = 0.0f;
  float m_time     = 0.0f;
};
```

`reinhard.cpp`
```cpp
#include "reinhard.h"
#include <algorithm>

static inline float reinhard_extended(float v, float max_white)
{
  float numerator = v * (1.0f + (v / (max_white * max_white)));
  return numerator / (1.0f + v);
}

void ReinhardTM::kernel1D_findMax(const float* hdrData, int size)
{
  whitePoint = 0.0f;                             // prologue: reduction init
  for(int i = 0; i < size; i++)
  {
    float r = hdrData[4*i+0], g = hdrData[4*i+1], b = hdrData[4*i+2];
    whitePoint = std::max(whitePoint, std::max(r, std::max(g, b)));   // max-reduction
  }
}

void ReinhardTM::kernel2D_process(int w, int h, const float* hdrData, uint32_t* ldrData)
{
  for(int y = 0; y < h; y++)
  {
    for(int x = 0; x < w; x++)
    {
      float r = reinhard_extended(hdrData[4*(y*w+x)+0], whitePoint);
      float g = reinhard_extended(hdrData[4*(y*w+x)+1], whitePoint);
      float b = reinhard_extended(hdrData[4*(y*w+x)+2], whitePoint);
      int ir = (int)std::min(r*255.0f, 255.0f);
      int ig = (int)std::min(g*255.0f, 255.0f);
      int ib = (int)std::min(b*255.0f, 255.0f);
      ldrData[y*w+x] = 0xFF000000 | (ib << 16) | (ig << 8) | ir;
    }
  }
}

void ReinhardTM::Run(int w, int h, const float* hdrData, uint32_t* ldrData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel1D_findMax(hdrData, w*h);                // w*h computed HERE, passed as a plain argument
  kernel2D_process(w, h, hdrData, ldrData);
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}
```

`whitePoint` is computed on the GPU by the first kernel, stays in GPU memory for the second
kernel, and is read back so the host can call `getWhitePoint()`.

## Multi-pass pipelines with intermediate buffers (bloom filter pattern)

All intermediates are member vectors sized by a host-side setter called before
`CommitDeviceData()`. Kernels receive them via `.data()`:

```cpp
void ToneMapping::SetMaxImageSize(int w, int h)      // host only
{
  m_width = w; m_height = h; m_widthSmall = w/4; m_heightSmall = h/4;
  m_brightPixels.resize(w*h);
  m_downsampledImage.resize(m_widthSmall*m_heightSmall);
  m_tempImage.resize(m_widthSmall*m_heightSmall);
}

void ToneMapping::Bloom(int w, int h, const float4* inData4f [[size("w", "h")]], unsigned int* outData1ui [[size("w", "h")]])
{
  kernel2D_ExtractBrightPixels(w, h, inData4f, m_brightPixels.data());
  kernel2D_DownSample4x(m_widthSmall, m_heightSmall, m_brightPixels.data(), m_downsampledImage.data());
  kernel2D_BlurX(m_widthSmall, m_heightSmall, m_downsampledImage.data(), m_tempImage.data());
  kernel2D_BlurY(m_widthSmall, m_heightSmall, m_tempImage.data(), m_downsampledImage.data());
  kernel2D_MixAndToneMap(w, h, inData4f, m_downsampledImage.data(), outData1ui);
}
```

The separable blur is two kernels (`BlurX`, `BlurY`) with a temporary buffer between them,
not one kernel with two loops. Filter weights live in a member `std::vector<float>` filled in
the constructor.

## Iterative algorithms

Plain `for` loops in the control function may call kernels repeatedly (Jacobi / Gauss-Seidel
style solvers, time steps, ping-pong between two member buffers):

```cpp
void Solver::Step(int w, int h, float* out_density [[size("w*h")]])
{
  for(int k = 0; k < STEPS_NUM; k++)
  {
    kernel2D_Diffuse(w, h, m_x0.data(), m_x1.data());   // x0 -> x1
    kernel2D_Diffuse(w, h, m_x1.data(), m_x0.data());   // x1 -> x0 (explicit ping-pong)
  }
  kernel1D_CopyOut(w*h, m_x0.data(), out_density);
}
```

## Service calls in control functions

```cpp
memcpy(a_outData, a_inData, a_size*sizeof(uint32_t));                     // GPU copy
std::exclusive_scan(a_data, a_data + a_size, a_outExc, 0);                // GPU prefix sum
std::inclusive_scan(a_data, a_data + a_size, a_outInc, std::plus<int>(), 0);
std::sort(a_out, a_out + a_size, [](uint2 a, uint2 b) { return a.x < b.x; });   // GPU sort
```

A control function is detected only if it calls at least one kernel.

## Textures

Pass `const Image2D<float4>&` (read) and `Image2D<float4>&` (write) to kernels and index
with `int2`; keep images as members of type `Image2D<float4>`. For filtered sampling keep a
`std::shared_ptr<ICombinedImageSampler>` member and call `m_tex->sample(uv)`. See
`apps/14_filter_bloom_textures` and `apps/tests/006_combined_image_sampler`.

## Other features

- `[[threadlocal]] float tmp[16];` for per-thread scratch arrays inside kernels or helpers.
- `[[kslicer::setter]] void SetOutput(MyInOut a_out);` lets the host hand over a struct of
  pointers that kernels use (see `apps/tests/004_setter`).
- Per-kernel options (work-group size, loop reorder) go into `kmake.json`, see
  [host_and_build.md](host_and_build.md).
