# Patterns, execution strategies and back ends

This page describes the main concepts. The current support status of every feature on every
back end is in [features.md](features.md).

## Architectural patterns

The pattern is chosen per kernel by its name prefix, and one class may mix them.
- **IPV** (Image Processing Vectorization): `kernel1D_/2D_/3D_` kernels with 1–3 nested `for` loops,
  i.e. a "parallel for". Use it for image filters, simulations, reductions, stencils, and also for
  ray marching or ray tracing with a bounded number of bounces.
- **RTV** (Ray Tracing Vectorization): `kernel_` kernels written as the code of one thread, which
  pass per-thread state (payload) between stages. Use it for path tracing and other multi-stage
  pipelines whose threads finish at different times.
- **BEV** (Block Expansion Vectorization), **experimental**: `kernelBE1D_` kernels for algorithms
  that work on blocks of data in a work group with shared memory. See below.

## Block Expansion Vectorization (experimental)

The kernel is a template on the block size. The outer loop runs over blocks, and each block becomes
one work group of `bsize` threads. Local arrays declared in the block body are placed in shared
memory. Loops marked `[[parallel]]` are executed by all threads of the work group, one iteration per
thread. Other statements are executed by one thread. Barriers are inserted between these parts
automatically.

```cpp
template<uint bsize>
void TestClass::kernelBE1D_Test(uint blockNum, float* out_buffer)
{
  for(uint blockId = 0; blockId < blockNum; blockId++)              // one work group per block
  {
    float blockData[bsize];                                          // shared memory

    for(int localId = 0; localId < bsize; localId++) [[parallel]]    // all threads of the group
      blockData[localId] = float(localId + blockId);

    blockData[5] = 5.0f;                                             // one thread

    for(int localId = 0; localId < bsize; localId++) [[parallel]]
      out_buffer[blockId*bsize + localId] = blockData[localId];
  }
}

// in a control function: the template argument sets the work group size
kernelBE1D_Test<64>(numElements/64, out_buffer);
```

Only 1D kernels are supported. The feature works with the Slang and GLSL back ends and is covered
by a single test, [`apps/tests/030_block_expansion_v1`](../apps/tests/030_block_expansion_v1).

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
