# RTV pattern (Ray Tracing Vectorization)

In RTV you write the algorithm **for a single thread**. The translator turns each per-thread
local variable of the control function into a GPU buffer (the "payload"), each `kernel_`
call into a compute dispatch over all threads, and early `break`/`return` into per-thread
"dead" flags checked by later kernels. Use it for multi-stage per-thread pipelines, typically
path tracing or per-particle processing with a variable number of iterations.

## Rules

1. Kernel names start with `kernel_` (no `1D`/`2D`). Their first arguments are the special
   thread-index names: `tid` for 1D, `tidX, tidY` for 2D (`tidZ` for 3D). The number of these
   arguments defines the grid dimension. The names are reserved: use exactly these.
2. A kernel body contains **no thread loop**; it processes one thread. Inner loops (for example
   over all spheres of a scene) are fine.
3. The control function has the same `tid` (or `tidX, tidY`) arguments. Its pointer arguments
   are annotated with the thread count: `[[size("tid")]]` or `__attribute__((size("tidX", "tidY")))`.
4. Inside the control function declare per-thread locals of POD or LiteMath types
   (`float4 rayPos; Lite_Hit hit; uint flags;`) and pass them to kernels **by pointer** (`&hit`).
   Kernels read them via `*ptr` and write via `*ptr = value`.
5. Global arrays (control-function arguments) are indexed in kernels with `tid`:
   `out_color[tid] = ...`.
6. A kernel may return `bool`. `if(!kernel_X(tid, ...)) break;` inside a loop, or
   `... return;` in the control function, marks the thread dead: later kernels in the loop are
   skipped for it, kernels after the loop still run.
7. For every RTV control function `Foo(uint tid, ...)` write a virtual CPU driver
   `FooBlock(uint tid, ..., uint a_passesNum)` that loops over all threads (OpenMP is fine).
   Its arguments are those of `Foo` plus `a_passesNum`; here `tid` means the **total thread
   count**. The generated class overrides `FooBlock` and runs the whole grid on the GPU.
   Host code calls `FooBlock`, never `Foo` directly.
8. Per-thread state that must persist across calls (random generators, accumulators) lives in
   member vectors indexed by `tid`, sized in the constructor from a `maxThreads` argument.
9. Options: `-megakernel 1` merges all kernels of a control function into one shader (often
   faster for short pipelines). It is a translator option, not a code change.

## Minimal example: SAXPY in RTV form (from `apps/tests/051_saxpy_ipv_rtv`)

```cpp
class Numbers
{
public:
  void SAXPY2(const float* a_adata  [[size("tid")]],
              const float* a_bdata  [[size("tid")]],
              const float* a_cdata  [[size("tid")]],
                    float* a_result [[size("tid")]], unsigned int tid);
  virtual void SAXPY2Block(const float* a_adata, const float* a_bdata, const float* a_cdata,
                           float* a_result, unsigned int a_size, unsigned int a_passesNum);

  void kernel_Mult2(float* a_res, const float* a_input1, const float* a_input2, unsigned int tid);
  void kernel_Add2 (float* a_res, const float* a_input1, const float* a_input2, unsigned int tid);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {}
};

void Numbers::SAXPY2Block(const float* a, const float* b, const float* c, float* r, unsigned int a_size, unsigned int a_passesNum)
{
  for(int tid = 0; tid < a_size; tid++)
    SAXPY2(a, b, c, r, tid);
}

void Numbers::SAXPY2(const float* a, const float* b, const float* c, float* r, unsigned int tid)
{
  float temp;                             // per-thread payload -> becomes a GPU buffer
  kernel_Mult2(&temp, a, b, tid);
  kernel_Add2(r, &temp, c, tid);
}

void Numbers::kernel_Mult2(float* a_res, const float* a_input1, const float* a_input2, unsigned int tid)
{
  *a_res = a_input1[tid] * a_input2[tid];
}

void Numbers::kernel_Add2(float* a_res, const float* a_input1, const float* a_input2, unsigned int tid)
{
  a_res[tid] = *a_input1 + a_input2[tid];
}
```

Note the difference to IPV: `a_res` is a per-thread payload in `kernel_Mult2` (`*a_res`) but a
global array in `kernel_Add2` (`a_res[tid]`), decided by what the control function passes.

## Path tracing example (from `apps/02_spheresStupidPt` and `03_spheresStupidPt_loopBreak`)

Header excerpt:

```cpp
class TestClass
{
public:
  TestClass(int a_maxThreads = 1) { InitSpheresScene(10); InitRandomGens(a_maxThreads); }

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY __attribute__((size("tidX", "tidY"))));
  void StupidPathTrace(uint tid, uint a_maxDepth,
                       const uint* in_pakedXY __attribute__((size("tid"))),
                       float4*     out_color  __attribute__((size("tid"))));

  virtual void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passesNum);
  virtual void StupidPathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passesNum);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);
  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);
  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar, Lite_Hit* out_hit);
  void kernel_InitAccumData(uint tid, float4* accumColor, float4* accumThoroughput);
  void kernel_NextBounce(uint tid, const Lite_Hit* in_hit, float4* rayPosAndNear, float4* rayDirAndFar,
                         float4* accumColor, float4* accumThoroughput);
  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color);

protected:
  float4x4                    m_worldViewProjInv;
  std::vector<float4>         spheresPosRadius;   // scene data, read by kernels
  std::vector<SphereMaterial> spheresMaterials;
  std::vector<RandomGen>      m_randomGens;       // per-thread state, indexed by tid
};
```

Control function with a per-thread loop and early exit:

```cpp
void TestClass::StupidPathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  kernel_InitAccumData(tid, &accumColor, &accumThoroughput);

  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  for(uint depth = 0; depth < a_maxDepth; depth++)
  {
    Lite_Hit hit;
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit))
      break;                                             // thread becomes inactive
    kernel_NextBounce(tid, &hit, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput);
  }

  kernel_ContributeToImage(tid, &accumColor, in_pakedXY, out_color);   // runs for all threads
}

void TestClass::StupidPathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passesNum)
{
  #pragma omp parallel for default(shared)
  for(uint i = 0; i < tid; i++)
    for(uint j = 0; j < a_passesNum; j++)
      StupidPathTrace(i, a_maxDepth, in_pakedXY, out_color);
}
```

Kernel that uses a per-thread random generator stored in a member vector:

```cpp
RandomGen gen = m_randomGens[tid];
const float2 uv = rndFloat2_Pseudo(&gen);
m_randomGens[tid] = gen;
```

Host code (constructor argument comes first in the factory):

```cpp
std::shared_ptr<TestClass> CreateTestClass_Generated(int a_maxThreads, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
...
pImpl = onGPU ? CreateTestClass_Generated(W*H, ctx, W*H) : std::make_shared<TestClass>(W*H);
pImpl->CommitDeviceData();
pImpl->PackXYBlock(W, H, packedXY.data(), 1);
pImpl->StupidPathTraceBlock(W*H, 6, packedXY.data(), realColor.data(), PASS_NUMBER);
```

## When RTV is the wrong choice

- A single stage over an array or image: use IPV, it is simpler.
- Ray tracing or ray marching with a bounded number of bounces that fits into one loop per
  pixel: use IPV and put the bounce loop inside the kernel body (see `apps/30_metaballs`).
- Stages that need data produced by *other* threads (neighbors, reductions): the RTV payload
  is private per thread, so use IPV kernels with member buffers between passes.
