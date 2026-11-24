# Patterns, execution strategies and back ends

This page describes the main concepts. The current support status of every feature on every
back end is in [features.md](features.md).

## Architectural patterns

There are two patterns of **control functions** and several patterns of **kernels**. The pattern of a kernel is
chosen by its name prefix; one class may contain kernels of all patterns.

### Patterns of control functions

- **IPV** (Image Processing Vectorization) is the main pattern. A control function is ordinary sequential code
  that calls kernels; every call becomes a dispatch (or a draw) on the GPU, in the same order. Kernels called from
  it are IPV, BEV or GPV kernels (see below).
- **RTV** (Ray Tracing Vectorization) is a more specific pattern. A control function is written as the code of one
  thread: it keeps per-thread state (payload) in local variables and passes it to `kernel_` kernels by pointer.
  Use it for path tracing and other multi-stage pipelines whose threads finish at different times. Only RTV control
  functions support megakernels and persistent rays ([below](#execution-strategies-from-one-source)).

### Patterns of kernels

| Pattern | Name prefix | A thread is | Called from | Typical use |
|---------|-------------|-------------|-------------|-------------|
| IPV | `kernel1D_`, `kernel2D_`, `kernel3D_` | an iteration of 1–3 nested `for` loops ("parallel for") | IPV control function | image filters, simulations, reductions, stencils, ray marching or ray tracing with a bounded number of bounces |
| BEV (Block Expansion Vectorization) | `kernelBE1D_`, `kernelBE2D_` | an iteration of a `[[parallel]]` loop inside a block; a block is a work group with shared memory | IPV control function | algorithms on blocks of data: block reduction and scan, tiled matrix multiplication, small neural networks for a batch of vectors |
| GPV (Graphics Pipeline Vectorization) | `kernelVS_`, `kernelPS_` | a vertex or a pixel | `LiteRS::Draw*` commands in an IPV control function | rasterization: vertex and pixel shaders with render targets and depth test |
| RTV | `kernel_` | the code of one thread, `tid` argument | RTV control function | multi-stage per-thread pipelines (path tracing) |

GPV: vertex and pixel shaders are const member functions written for one vertex or one pixel; the pipeline state is a
`LiteRS::Pipeline` member of the class (`apps/LiteMath/raster.h`). See [features.md](features.md#graphics-pipeline)
for the supported features and samples.

## Block Expansion Vectorization (BEV, experimental)

A BEV kernel is a template on the size of the block. The outer loop (two loops for `kernelBE2D_`) runs over
blocks, and each block becomes one work group. The code of the block is executed as follows.

- **Arrays declared at the top level of the block** are placed in shared memory; they must not have initializers.
- **A `[[parallel]]` loop** is distributed over the threads of the group: thread `t` executes iterations `t`,
  `t + bsize`, ... A loop may have more iterations than the group has threads. Barriers are added before and after it.
- **Ordinary loops and `if` with `[[parallel]]` loops inside** are executed by all threads of the group, so their
  conditions must be the same for all threads: they may depend on the number of the block, arguments, constants,
  variables of the block and shared memory, but not on variables of parallel loops.
- **Other statements that write memory** (arrays of the block, buffers, members of the class, arguments passed by
  pointer or reference) are executed by one thread; a barrier is added if they write shared memory.
- **All other statements** (local variables of the block, conditions, reading memory) are executed by every thread,
  and every thread has its own copy of the variables of the block. So a variable of the block can be changed in an
  ordinary loop and read in a parallel loop without shared memory.

Example: a block sums 256 elements (`apps/tests/092_bev_block_reduce`).

```cpp
template<uint bsize>
void BlockReduce::kernelBE1D_ReduceBlocks(uint a_blockNum, const int* a_in, int* a_out, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)   // one work group per block
  {
    int sdata[bsize];                                     // shared memory

    for(uint i = 0; i < bsize; i++) [[parallel]]          // all threads of the group
    {
      const uint globalId = blockId*bsize + i;
      sdata[i] = (globalId < a_size) ? a_in[globalId] : 0;
    }

    for(uint stride = bsize/2; stride > 0; stride /= 2)   // all threads, barriers inside
    {
      for(uint i = 0; i < stride; i++) [[parallel]]
        sdata[i] += sdata[i + stride];
    }

    a_out[blockId] = sdata[0];                            // writes memory: one thread
  }
}

// in a control function: the template argument is the size of the work group, a named constant may be used
kernelBE1D_ReduceBlocks<BLOCK_SIZE>(blocks, a_in, m_sums.data(), a_size);
```

**2D blocks.** `kernelBE2D_` has two template parameters, the sizes of the group along X and Y, and two loops over
blocks. A 2D parallel loop is two perfectly nested `[[parallel]]` loops; arrays of the block may be two-dimensional.
Dimensions are assigned to loops in the same way as for `kernel2D_`: without options X is the outer loop, with
`-reorderLoops YX` it is the inner one. Write loops for the CPU (`y` outer, `x` inner) and use `-reorderLoops YX`
(`apps/tests/096_bev_tiled_gemm_2d`):

```cpp
template<uint bsizeX, uint bsizeY>
void TiledGemm2D::kernelBE2D_MatMul(uint a_blocksX, uint a_blocksY, ...)
{
  for(uint blockY = 0; blockY < a_blocksY; blockY++)
  {
    for(uint blockX = 0; blockX < a_blocksX; blockX++)
    {
      float tileA[bsizeY][bsizeX];                        // shared memory
      ...
      for(uint y = 0; y < bsizeY; y++) [[parallel]]       // thread (x, y) of the group
        for(uint x = 0; x < bsizeX; x++) [[parallel]]
          tileA[y][x] = ...;
    }
  }
}
// kernelBE2D_MatMul<16, 16>(blocksX, blocksY, ...)
```

A single `[[parallel]]` loop in a 2D block is distributed over all `bsizeX*bsizeY` threads.

**Block level functions.** `LiteNN::LinearForwardBlock` (`apps/LiteMathAux/neural.h`) computes a fully connected
layer for a batch of vectors stored in arrays of the block: fp16 inputs, fp32 accumulation and outputs. It is called
from the code of the block (not from a parallel loop) and executed by all threads of the group together. By default
it is an ordinary loop; with `-coop_matrix 1` the Slang back end uses cooperative matrices 16x16x16
(`VK_KHR_cooperative_matrix`). See `apps/tests/095_bev_mlp_batch`.

**Restrictions.**
- A statement that writes memory and also changes a variable of the block is an error: it is executed by one thread,
  so the copies of the variable in other threads would differ. Split it into two statements.
- `[[parallel]]` inside `[[parallel]]` is allowed only as a 2D parallel loop of `kernelBE2D_`; there is no 3D.
- Barriers synchronize shared memory only: data written to a buffer by one thread of the group must not be read by
  other threads of the same group in the same kernel.
- `static_assert` in the code of the block is checked by the C++ compiler and skipped by the translator.
- Atomics: `InterlockedAdd` on arrays of the block (shared memory) and on buffers works in parallel loops
  (`apps/tests/097_bev_histogram`).
- Back ends: Slang, GLSL (without `LiteNN` functions) and ISPC. In ISPC the code of the block is uniform, parallel
  loops are `foreach`, and functions called from the code of the block get uniform overloads.

Tests: `apps/tests/030_block_expansion_v1`, `092_bev_block_reduce`, `093_bev_block_scan`, `094_bev_tiled_gemm`,
`095_bev_mlp_batch`, `096_bev_tiled_gemm_2d`, `097_bev_histogram`.

## Algorithmic patterns

The translator recognizes these patterns in the code:
- parallel reduction to a class member (sum, min, max), including over arrays;
- prefix sum (`std::exclusive_scan`, `std::inclusive_scan`) and sorting (`std::sort` with a lambda);
- `memcpy` between buffers;
- appending to an array (`m_vec.push_back` inside a kernel) and indirect dispatch whose size is
  computed on the GPU;
- atomics, including float atomics, with a CAS emulation option for GPUs that lack them;
- hash tables (`std::unordered_map` fields of the class);
- textures and samplers;
- virtual functions;
- ray tracing queries through an abstract scene interface.

## Cross-platform hardware acceleration for ray tracing

User code traces rays through an abstract scene interface
([CrossRT](https://github.com/msu-graphics-group/CrossRT), the `apps/RayTracing` submodule). On the
CPU it may be implemented with Embree. On the GPU the translator generates either Vulkan hardware
ray tracing or a software BVH traversal compiled from C++. Custom primitives (SDF, NURBS and others)
are defined by a C++ intersection function, which is embedded into both traversal variants.

## Execution strategies from one source

Strategies are selected with translator options: one megakernel (`-megakernel 1`) or a set of
separate kernels (wavefront) with automatic layout of per-ray data in memory. These variants
differ in their kernels, resources and host code. Maintaining them by hand is impractical; here
you switch an option.

**Persistent rays** (`-persistent 1`, with `-megakernel 1`) reorder rays for Monte Carlo sampling:
all threads of a subgroup trace samples of the same pixel, which reduces thread and data
divergence. One megakernel call computes `RTVPersistent_Iters()` samples per pixel (the subgroup
size). The user marks the place where the result is written to the image with dummy member
functions of the main class:

```cpp
colorRes = ...;                                   // contribution of the sample
colorRes = RTVPersistent_ReduceAdd4f(colorRes);   // sum over the subgroup
if(RTVPersistent_IsFirst())                       // only the first thread of the subgroup
{
  ...                                             // writes 'colorRes' to the image
}
```

Without the option `RTVPersistent_IsFirst()` returns `true` and `RTVPersistent_ReduceAdd4f`
(`RTVPersistent_ReduceAdd1f` for `float`) returns its argument. With it, they become
`WaveGetLaneIndex() == 0` and `WaveActiveSum`, and the host code must launch the megakernel
`RTVPersistent_Iters()` times fewer. Do not confuse this with the general *persistent threads*
technique.

**On the CPU (ISPC back end)** both strategies are available too, persistent rays are not. With
`-megakernel 1` the control function becomes one ISPC function of a thread; payload (local variables
of the control function passed to kernels by pointer) are ISPC references, and OpenMP threads process
chunks of `-ispc_mega_chunk` threads. With `-megakernel 0` the control function runs on the host for
chunks of `-ispc_wave_chunk` threads: each kernel processes all threads of the chunk before the next
one, payload is stored in arrays of the chunk (one set of arrays for each OpenMP thread, small enough
to stay in cache), finished threads are skipped by thread flags as on the GPU.

## Back ends

The back end is selected with `-shaderCC`:

| Value | Output | Status |
|-------|--------|--------|
| `slang` | Vulkan host code + Slang shaders (`slangc` → SPIR-V) | **main, recommended** |
| `glsl`  | Vulkan host code + GLSL shaders (`glslangValidator`) | legacy, maintained |
| `cuda` / `hip` / `musa` | CUDA, HIP, MUSA | experimental |
| `ispc`  | CPU vectorization with Intel ISPC | experimental |
| `wgpu`  | WebGPU | experimental |
| `clspv` | OpenCL C → SPIR-V via clspv (the built-in default) | outdated, always set `-shaderCC` explicitly |
