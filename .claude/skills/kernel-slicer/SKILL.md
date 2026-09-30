---
name: kernel-slicer
description: Write restricted C++ for kernel_slicer (kslicer), the source-to-source translator that turns an ordinary C++ class into a GPU implementation (Vulkan with Slang shaders by default, legacy GLSL, also CUDA, ISPC, WebGPU). Use whenever the user asks to implement an algorithm "with/on kernel_slicer", to port C++ code to GPU via kslicer, to write kernel1D_/kernel2D_/kernel_ functions or control functions with [[size(...)]] annotations, to choose between the IPV and RTV patterns, or to create main.cpp, CMakeLists.txt or kmake.json for a kernel_slicer sample.
---

# kernel_slicer programming skill

kernel_slicer reads **one C++ class** (the *main class*) and generates a derived class
`<MainClass>_Generated` that runs the same algorithm on the GPU. You write and debug plain
CPU C++; the translator does the port. Your job is to write the CPU code so that it fits
the patterns the translator recognizes. Code that does not fit is silently ignored or
produces broken shaders, so follow the rules below literally.

## Mental model

| Role | How it is recognized | Runs on GPU as |
|------|----------------------|----------------|
| **Kernel function** | member function whose name starts with `kernel1D_`, `kernel2D_`, `kernel3D_` (IPV) or `kernel_` (RTV) | one compute shader |
| **Control function** | member function that calls at least one kernel | host code: records a Vulkan command buffer that launches the kernels in order |
| **Helper function** | any free or member function called from a kernel | copied into shader code |
| **Data members** | POD members and `std::vector<POD>` members of the main class | uniform buffer / storage buffers |

Call graph is strictly two-level: host code → control function → kernel → helpers.
- A kernel must not call another kernel.
- A control function must not call another control function.
- Both may call ordinary helper functions.

## Step 1. Choose the pattern (IPV is the default)

The pattern is chosen **per kernel by its name prefix**; one class may mix both.

- **IPV, Image Processing Vectorization** (`kernel1D_`, `kernel2D_`, `kernel3D_`). The kernel body
  contains 1, 2 or 3 perfectly nested `for` loops; these loops *are* the thread grid. This is the
  classic "parallel for". Use it for image filters, array maps, reductions, stencils, matrix ops,
  simulations over grids, particles and nearly everything else.
- **RTV, Ray Tracing Vectorization** (`kernel_`). The kernel body has **no** thread loop; it is the
  code of one thread, identified by the special arguments `tid` (1D) or `tidX, tidY[, tidZ]`.
  The control function is also written per thread, keeps per-thread locals (the "payload") and
  passes them to kernels by pointer. Use it when every thread runs a *pipeline of stages with
  per-thread state*, often inside a loop that threads leave at different times (`break`/`return`):
  path tracing, per-ray or per-particle multi-stage processing.

A bounded per-pixel loop does **not** require RTV. Ray marching or ray tracing with a fixed maximum
number of reflections fits into one IPV kernel: the bounces are an ordinary loop inside the
kernel body (`apps/30_metaballs` renders 1024×1024 with 3 reflections in about 7 ms this way).
Choose RTV only when the per-thread work should be split into separate kernels that pass
per-thread state between them, for example a long path-tracing loop with early exit.

If unsure, pick IPV. Full rules and a worked example: [references/ipv.md](references/ipv.md),
[references/rtv.md](references/rtv.md).

## Step 2. Class skeleton (both patterns)

```cpp
#pragma once
#include <vector>
#include <cstdint>
#include "LiteMath.h"          // HLSL-like types: float2..4, int2..4, uint2..4, float3x3, float4x4
using LiteMath::float4;        // import every LiteMath type/function you use
using LiteMath::uint;

class MyAlgo                   // the "main class"; pass its name as -mainClass
{
public:
  MyAlgo() {}
  void SetMaxSize(int w, int h);          // host-only: size member vectors BEFORE CommitDeviceData()

  // control function: virtual, pointer args annotated with size, const = input, non-const = output
  virtual void Run(int w, int h, const float4* a_in [[size("w*h")]], uint32_t* a_out [[size("w*h")]]);

  // mandatory, overridden in the generated class
  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; }

protected:
  virtual void kernel2D_Stage1(int w, int h, const float4* a_in, float4* a_tmp);
  virtual void kernel2D_Stage2(int w, int h, const float4* a_tmp, uint32_t* a_out);

  std::vector<float4> m_tmp;              // every intermediate buffer is a class member
  int   m_width = 0, m_height = 0;
  float m_time  = 0.0f;
};
```

## Step 3. Control function rules

1. Every pointer argument carries a size annotation: `[[size("expr")]]` or
   `__attribute__((size("expr")))`. `expr` uses the function's own scalar arguments, e.g.
   `"a_size"`, `"w*h*4"`, `"2"`. Several arguments mean a multi-dimensional size:
   `__attribute__((size("w", "h")))`.
2. `const T*` means **input** (copied CPU→GPU). Non-const `T*` means **output** (copied GPU→CPU).
   In-out pointers are not allowed: use a separate const input and non-const output.
3. Do not declare `std::vector` (or any heap container) inside a control function. Every
   temporary buffer becomes a member vector, sized in the constructor or a setter, and is
   passed to kernels as `m_vec.data()`.
4. The body is a sequence of kernel calls, optionally inside plain `for` loops (iterations,
   ping-pong steps) with scalar arithmetic on arguments. Timing with `std::chrono` around the
   calls is allowed and conventional.
5. Allowed service calls between kernels: `memcpy(dst, src, bytes)`,
   `std::exclusive_scan` / `std::inclusive_scan`, `std::sort`. They are replaced by GPU
   implementations.
6. No `throw`, no I/O, no allocation, no calls to other control functions.
7. Small per-call parameter arrays (filter weights, coefficients, a transform matrix list) are
   best passed as extra `const T* [[size("n")]]` arguments: they are uploaded on every call.
   Member vectors reach the GPU only through `CommitDeviceData()`, so changing them between
   calls requires another commit.
8. Scalar class members written by kernels (e.g. a reduction result) are read back to the CPU
   after the control function returns, so host code can read `pImpl->m_summ`.

## Step 4. Kernel and helper code rules

**Allowed**
- Arithmetic, `if`/`switch`, inner (non-thread) `for`/`while` loops with bounded trip counts,
  including `break` out of an inner loop.
- `bool` locals, and locals declared without an initializer and assigned in `if`/`else` branches.
- `const` member helper functions that read POD and `std::vector` members, with a class member
  or a `static constexpr` class constant as the loop bound.
- LiteMath types and functions: `dot`, `cross`, `normalize`, `length`, `clamp`, `min`, `max`,
  `lerp`, `to_float3`, `make_float4`, matrix × vector, etc.; `std::min`, `std::max`, `std::abs`,
  `std::sqrt`, `std::exp`, `std::log`, `std::pow`, `std::sin`, `std::floor`, ... Only what
  `TINYSTL/cmath` declares works in `std::` form: there is **no `std::fabs`**, use `std::abs`.
- Reading any POD member and any `std::vector` member (`m_vec[i]`, `m_vec.size()`,
  `m_vec.capacity()`, `m_vec.data()`).
- `m_vec.push_back(x)` and `m_vec.resize(0)` on member vectors (capacity must be reserved on the
  host beforehand).
- Calling non-kernel member functions and free `inline`/`static` functions.
- `#pragma omp parallel for` on the outer thread loop: it speeds up the CPU version and the
  translator ignores it.
- Small fixed-size local arrays; mark large ones `[[threadlocal]] float tmp[64];`.
- Structs of POD fields, constants via `constexpr` / `static constexpr` / `#define`.
- Atomics: `LiteMath::InterlockedAdd(a_out[idx], value)` for `int`/`uint`/`float`.

**Forbidden inside kernels and helpers**
- Calling another kernel; recursion; virtual calls (except the special advanced samples).
- Local `std::vector`, `std::string`, `new`/`malloc`, `std::function`, lambdas with captures,
  exceptions, `printf`/`std::cout`/file I/O.
- Pointer class members (`float* m_ptr` is ignored by the translator), nested containers
  (`std::vector` inside a struct stored in the main class), comparing pointers with `nullptr`.
- `double`, unless you know the device supports it; write float literals as `1.0f`.
- **Class members or kernel arguments directly inside a `std::` call** (`std::max(m_val, x)`,
  `std::pow(x, a_gamma)`). The Slang back end truncates them (`m_val` becomes `ubo[0`) and the
  shader does not compile. Copy them to a local first:
  `const float val = m_val; ... std::max(val, x)`. Locals and helper-function parameters inside
  `std::` calls are fine, and so is the reduction statement `m_max = std::max(m_max, v);`.
- Two or more consecutive top-level loops in one IPV kernel: split them into two kernels.

**Data layout**
- Prefer `float4` over `float3` inside structs and buffers; keep struct sizes a multiple of 16
  bytes when they contain vector types, adding padding fields if needed (translator warning A1).
- Keep the host pointer for `float4*` data 16-byte aligned.

## Step 5. Host code and build

Summary; the full templates are in [references/host_and_build.md](references/host_and_build.md).

```cpp
std::shared_ptr<MyAlgo> CreateMyAlgo_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures MyAlgo_Generated_ListRequiredDeviceFeatures();
...
std::shared_ptr<MyAlgo> pImpl = onGPU ? CreateMyAlgo_Generated(ctx, w*h) : std::make_shared<MyAlgo>();
pImpl->SetMaxSize(w, h);        // resize/reserve member vectors first
pImpl->CommitDeviceData();      // then upload class data to GPU
pImpl->Run(w, h, in.data(), out.data());
```

- If the constructor has arguments, they come first in the factory:
  `CreateMyAlgo_Generated(ctorArg1, ..., ctx, maxThreads)`.
- `a_maxThreadsGenerated` is the largest thread count any kernel will use.
- For RTV, the host calls the `...Block` function, not the per-thread one.
- Generated files are named after the **source file**: `conv2d.cpp` gives `conv2d_generated.h`,
  `conv2d_generated.cpp`, `conv2d_generated_ds.cpp`, `conv2d_generated_init.cpp` and the
  folder `shaders_generated/`. Class and factory names come from the **class** name.
- After every kslicer run, recompile the shaders (`shaders_generated/build_slang.sh`) and
  rebuild the application.

Translate with kslicer from the repository root (elsewhere add `-selfdir <repo root>`):

```bash
./cmake-build-release/kslicer apps/my_app/my_algo.cpp -mainClass MyAlgo \
  -stdlibfolder TINYSTL -Iapps/LiteMath ignore -Iapps/LiteMathAux ignore -ITINYSTL ignore \
  -shaderCC slang -DKERNEL_SLICER -v
cd apps/my_app/shaders_generated && bash build_slang.sh   # slangc -> *.spv
```

Always pass `-shaderCC slang`. It is the current back end; `glsl` is legacy (its script is
`build.sh`), and the built-in default is the even older clspv.

## Step 6. Workflow for a new task

1. Restate the algorithm as a sequence of data-parallel passes. Every pass is one kernel.
   Every "for each element, then for each element again" becomes two kernels.
2. Pick IPV or RTV.
3. Write the header: main class, control function(s) with size annotations, kernels,
   member vectors for all intermediates, `CommitDeviceData`, `GetExecutionTime`.
4. Write kernels, helpers and control functions in one `.cpp` (the file passed to kslicer).
5. Write `main.cpp` with a `--gpu` switch that selects CPU or generated implementation, and a
   `--compare` mode that creates both implementations in one process, runs the same inputs
   and reports the difference. Return a non-zero exit code on mismatch. Choose the metric by
   the kind of algorithm:
   - Direct arithmetic such as filters or maps: maximum relative difference, typically about
     1e-7; use a tolerance such as 1e-4.
   - Ray marching, root finding and other threshold-sensitive code: a few pixels on silhouettes
     can differ strongly because of the last float bit. Count pixels that differ by more than
     2/255 and require their share to stay small, for example below 0.5%.
   - Sum reductions over millions of elements: the GPU adds in a different order, so compare
     with a relative tolerance of about 1e-3, not bit for bit.
6. Write `CMakeLists.txt` (and optionally `kmake.json`) from the template.
7. Build and run the CPU version first (`cmake -DUSE_VULKAN=OFF`, the generated files are not
   needed). Then run kslicer, compile shaders, build and run `--compare`.
8. kslicer returns exit code 0 even when it failed. Check two places: its log for `error:`,
   and the output of `build_slang.sh`, where a pattern violation often surfaces only as a
   compile error in a generated shader. A failed shader leaves no `.spv`, and the application
   then fails at start with `can't open file shaders_generated/....spv`.

## Pre-delivery checklist

- [ ] Kernel names have exactly one prefix: `kernel1D_`/`kernel2D_`/`kernel3D_` (IPV) or `kernel_` (RTV).
- [ ] IPV: the number of nested thread loops equals the `ND` in the name; loops are perfectly
      nested with nothing between them; there is only one such loop nest per kernel.
- [ ] IPV: every thread-loop bound is a plain kernel argument or class member. An expression
      such as `i < w*h` is accepted silently but produces uncompilable shaders.
- [ ] IPV: code before and after the loop nest only initializes/finalizes members or writes
      single values (it runs once, not per thread).
- [ ] RTV: first kernel arguments are `tid` or `tidX, tidY`; no thread loop in the body;
      a `...Block(..., uint a_passesNum)` virtual function exists for each RTV control function.
- [ ] No kernel calls a kernel; no control function calls a control function.
- [ ] Every control-function pointer has `[[size(...)]]`; inputs are `const`; no in-out pointers.
- [ ] No `std::vector` declared inside control functions or kernels; all buffers are members
      sized before `CommitDeviceData()`.
- [ ] Only LiteMath vector types; `using LiteMath::...` for each one used.
- [ ] `CommitDeviceData()` and `GetExecutionTime()` are declared virtual in the class.
- [ ] No I/O, exceptions, allocation, recursion or pointer members in GPU-side code.
- [ ] `std::` math calls exist in `TINYSTL/cmath` (no `std::fabs`, `std::fmin`, `std::fmax`).
- [ ] `kmake.json`, if any, sets `wgSize` only for named kernels, never a 2D size under `"default"`.
- [ ] No class member or kernel argument written directly inside a `std::` call in GPU code.
- [ ] kslicer log has no `error:`; `build_slang.sh` compiled every shader; `--compare` passes.

Common mistakes with corrected versions: [references/pitfalls.md](references/pitfalls.md).

## Reference samples in this repository

Look only at the hand-written files (`main.cpp`, `test_class.h`, `test_class.cpp` or equivalent).
Files with `_generated`, `_gpu`, `_ispc` suffixes and `shaders_*` folders are translator output.

| Topic | Sample |
|-------|--------|
| Minimal IPV, reduction to a member | `apps/04_array_summ` |
| **Complete verified IPV app**: 2D/separable convolution 3×3–7×7, reduction, kmake.json, `--compare` | `apps/29_conv2d` |
| **Ray tracing in IPV style** (no RTV): metaballs, reflections, shadows, auto exposure; VS Code configs | `apps/30_metaballs` |
| Multi-kernel image pipeline, member buffers | `apps/05_filter_bloom_good`, `apps/24_reinhard_tm` |
| `push_back` into a member vector, `memcpy` in control function | `apps/08_push_back_red_pixels` |
| Iterative solver, loops of kernel calls | `apps/19_simple_2d_cfd_solver` |
| Textures (`Image2D`, sampler) | `apps/14_filter_bloom_textures`, `apps/tests/006_combined_image_sampler` |
| RTV path tracer | `apps/02_spheresStupidPt`, `apps/03_spheresStupidPt_loopBreak`, `apps/07_simple_pt` |
| Same algorithm in IPV and RTV side by side | `apps/tests/051_saxpy_ipv_rtv` |
| Reductions (min/max/sum), bounding box | `apps/tests/003_reduction`, `apps/tests/004_setter` |
| Atomics, `ReduceAdd` | `apps/tests/041_atomic_add_int`, `042_atomic_add_float`, `045_reduce_add` |
| Scan and sort service calls | `apps/tests/023_prefix_summ`, `apps/tests/025_sort_v1` |
| `[[threadlocal]]` arrays | `apps/tests/033_threadlocal_array_v1` |
| `[[kslicer::setter]]` | `apps/tests/004_setter` |
