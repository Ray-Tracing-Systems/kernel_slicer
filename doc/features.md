# Feature status

This page shows what the translator supports on each back end and where to see an example. It is
kept up to date by the developers: when you add, fix or break a feature, update its row (see
[Maintaining this page](#maintaining-this-page)).

Every feature has a short **ID**. IDs are stable: they are used to refer to features from tests,
configs and issues. Do not rename an ID; if a feature is split, add new IDs and keep the old one.

A program may list the features it needs in its `kmake.json`:

```jsonc
"requireFeatures" : ["scan", "rt_query_hw"]
```

kslicer reads this page at run time and stops with an error if the selected back end does not
support one of the features (❌) or its status is unknown (❔). For 🟡 and 🧪 it prints a warning.
Without `requireFeatures` nothing is checked. If this page is not found (kslicer is started outside
the repository without `-selfdir`), kslicer prints a warning and continues. Because kslicer parses
this page, the format of the tables (the `ID` column, back end column names, marks) must be kept.

| Mark | Meaning |
|------|---------|
| ✅ | supported, covered by tests |
| 🟡 | partially supported, see the note |
| 🧪 | experimental: works in the listed tests, may change or break |
| ❌ | not supported |
| ❔ | unknown: not implemented on purpose, not tested |

Back ends: **Slang** (`-shaderCC slang`, main), **GLSL** (`-shaderCC glsl`, legacy), **CUDA/HIP**
(`-shaderCC cuda`, `hip`, `musa`, experimental), **ISPC** (`-shaderCC ispc`, CPU, experimental).
WebGPU and clspv are not listed: they are too experimental or outdated. See
[concepts.md](concepts.md#back-ends).

Tests are in `apps/tests/NNN_*` (written below as `NNN`) and samples in `apps/NN_*` (written as
`app NN`).

Last full check: 2026-11-05, `python3 ktest.py -b slang,glsl,ispc -s all` on AMD Radeon 8060S
(RADV). Every Slang and GLSL mark is backed by a variant in the `ktest.json` of the listed tests;
known failures are variants with `"skip"`. The CUDA/HIP column was not checked on hardware: it
follows the test configs and the code.

## 1. Functional and hardware features

### Patterns and execution

| ID | Feature | Slang | GLSL | CUDA/HIP | ISPC | Tests and samples |
|----|---------|:-----:|:----:|:--------:|:----:|-------------------|
| `ipv` | IPV kernels `kernel1D_/2D_/3D_` (parallel `for`) | ✅ | ✅ | ✅ | ✅ | 048, 043, app 05, app 25, app 30 |
| `rtv` | RTV kernels `kernel_` (code of one thread, payload between kernels) | ✅ | ✅ | ❔ | ❔ | 051, 067, app 01, app 07 |
| `bev` | Block Expansion kernels `kernelBE1D_` with shared memory | 🧪 | 🧪 | ❔ | ❔ | 030 |
| `megakernel` | One shader per control function (`-megakernel 1`) | ✅ | ✅ | ❔ | ❌ | 057, 067, app 07, app 20 |
| `persistent_rays` | Persistent rays in RTV megakernels (`-persistent 1`) | 🧪 | 🧪 | ❔ | ❔ | — ⁰ |
| `indirect_dispatch` | Dispatch size computed on GPU (loop bound is a class member) | ✅ | ✅ | ❔ | ❔ | 059, 057 |
| `reorder_loops` | `-reorderLoops YX` for 2D kernels | ✅ | ✅ | ✅ | ❔ | 059, app 05, app 24 |
| `subgroups` | Subgroup operations in reductions (`-enableSubgroup`) | ✅ | ✅ | ✅ | ❔ | app 04, 055 |
| `timestamps` | GPU time of kernels, `GetExecutionTime` (`-timestamps 1`) | ✅ | ✅ | ❔ | ❔ | app 05, app 28 |
| `setters` | Setter functions that pass GPU objects to the class | ✅ | ✅ | ❔ | ❔ | 004, 005 |
| `composition` | Main class with composed classes (`composClasses`) | ✅ | ✅ | ❔ | ❔ | 018, 019 |
| `update_plain_data` | `UpdateMembersPlainData()`: upload only POD members | ✅ | ✅ | ❔ | ❔ | 068 |
| `commit_appended` | `CommitAppendedData()`: upload only appended elements (`appendGroups`) | ✅ | ✅ | ❔ | ❔ | 064, 065 |

⁰ No test yet. See [concepts.md](concepts.md#execution-strategies-from-one-source).

### Algorithms and service functions

| ID | Feature | Slang | GLSL | CUDA/HIP | ISPC | Tests and samples |
|----|---------|:-----:|:----:|:--------:|:----:|-------------------|
| `reduction` | Reduction to a class member (`+=`, `min`, `max`), also over arrays | ✅ | ✅ | ✅ | ✅ | 003, app 04 |
| `reduce_add` | `LiteMath::ReduceAdd` into a buffer | ✅ | ❌ | ✅ | ❔ | 045 |
| `scan` | `std::exclusive_scan`, `std::inclusive_scan` | ✅ | ✅ | ❌ | ❔ | 023, 024 |
| `sort` | `std::sort` with a lambda comparator | ✅ | ✅ | ❌ | ❔ | 025, 026 |
| `memcpy` | `memcpy` between buffers in control functions | ✅ | ✅ | ❔ | ❔ | 057 |
| `append` | `m_vec.push_back(x)` inside kernels | ✅ | ✅ | ✅ | ❔ | app 08 |
| `atomics_int` | `InterlockedAdd` on `int`/`uint` | ✅ | ✅ | ❔ | ❔ | 041 |
| `atomics_float` | Float atomics (hardware `shaderBufferFloat32AtomicAdd`) | ✅ | ✅ | ✅ | ❔ | 042, 054, 045 |
| `atomics_float_emul` | Float atomics emulated with CAS (`-atomicf_emul 1`) | ✅ | ✅ | ❔ | ❔ | 042, 054, 056 |
| `hash_map` | `std::unordered_map` members | ✅ | ❌ | ❔ | ❔ | 055, 056 |
| `nn_coop_vec` | Small neural networks with cooperative vectors (`LiteNN`) | 🧪¹ | ❌ | ❌ | ❌ | 071, 072, 073 |

¹ Needs `VK_NV_cooperative_vector` (NVIDIA). Not available on the GPU of the last check.

### Data, memory and textures

| ID | Feature | Slang | GLSL | CUDA/HIP | ISPC | Tests and samples |
|----|---------|:-----:|:----:|:--------:|:----:|-------------------|
| `ubo_layout` | POD members of any types and order in the UBO (padding) | ✅ | ✅ | ❔ | ❔ | 009–015, 021 |
| `fp64` | `double` | ✅ | ✅ | ❔ | ❔ | 016, 017 |
| `fp16` | `half` arithmetic and conversions | 🟡² | 🟡³ | ❔ | ❔ | 022, 053, 070 |
| `storage_8_16bit` | 8/16-bit types in buffers and UBO | ✅ | ✅ | ❔ | ❔ | 060, 053 |
| `buffer_reference` | Buffer device address (`-with_buffer_reference`) | ❌ | ✅ | ❔ | ❔ | 039 |
| `textures` | Textures and samplers (`ITexture2D`, combined image sampler) | 🟡⁴ | ✅ | ❌ | ❔ | 006, 007, 008, app 14, app 15, app 20 |
| `image_storage` | Writable images (`Texture2D<float4>` output) | ❌ | ✅ | ❌ | ❔ | 066 |
| `half_textures` | Textures stored in half precision (`-halfTex 1`) | ❌ | ✅ | ❌ | ❔ | 066, app 14 |

² 053 and 070 work, 022 (`-forceEnableHalf`) fails to compile.
³ 022 and 070 work, 053 (`as_half`/`bit_cast`) fails to compile.
⁴ Arrays of textures (007) work. A single combined image sampler member (006, 008) and the texture
samples app 14, app 15 fail to compile.

### Ray tracing and virtual functions

| ID | Feature | Slang | GLSL | CUDA/HIP | ISPC | Tests and samples |
|----|---------|:-----:|:----:|:--------:|:----:|-------------------|
| `rt_query_hw` | `ISceneObject::RayQuery_*` with hardware ray queries | ✅ | ✅ | ❌ | 🧪⁵ | 019, 065, app 16, app 18 |
| `rt_bvh_sw` | Software BVH traversal compiled from C++ | ✅ | ✅ | ❌ | ❔ | 046, 065 |
| `rt_pipeline` | Vulkan ray tracing pipeline (`-enable_ray_tracing_pipeline 1`) | ❌ | ✅ | ❌ | ❌ | 035, 036, 037, app 17 |
| `rt_custom_prim` | Custom primitives with a C++ intersection function | 🟡⁶ | ✅ | ❌ | ❔ | 046, 035–038 |
| `rt_instances` | Instanced scenes | 🟡⁷ | ✅ | ❌ | ❔ | app 16, app 17 |
| `rt_scene_append` | Adding meshes and instances to a scene on the fly | ✅ | ✅ | ❌ | ❔ | 065 |
| `virtual_func` | Virtual function hierarchies in kernels | ❌ | ✅ | ❔ | ❔ | 069, app 10, app 13 |
| `callable_shaders` | Virtual functions as callable shaders (`-enable_callable_shaders 1`) | ❌ | ✅ | ❌ | ❌ | app 26, app 27 |

⁵ With Embree on the CPU, IPV kernels only: app 18.
⁶ The current interface (046) works; the older `-intersectionShader` configs (035–038) fail to compile.
⁷ app 16 works, app 17 needs `rt_pipeline`.

## 2. C++ language features

These rules apply to kernels and to the functions they call. On the host side (control functions
and other code) any C++ is allowed.

| ID | Feature | Slang | GLSL | CUDA/HIP | ISPC | Tests and samples |
|----|---------|:-----:|:----:|:--------:|:----:|-------------------|
| `member_func` | Calling non-kernel member functions of the main class | ✅ | ✅ | ✅ | 🧪 | 044, app 18 |
| `free_func` | Free `static`/`inline` functions, forward declarations | ✅ | ✅ | ❔ | ❔ | 058 |
| `struct_pod` | Structs of POD fields, aggregate initialization | ✅ | ✅ | ❔ | ❔ | 020 |
| `struct_member_func` | Member functions of user structs (`const`, modifying, `static`) | ✅ | ❌ | ❔ | ❔ | 061 |
| `constructors` | Constructors of user structs, initializer lists | ✅ | ❌ | ❔ | ❔ | 040 |
| `default_member_init` | Default member initializers of structs | ✅ | ✅ | ❔ | ❔ | 075 |
| `inheritance` | The main class derived from base classes (`baseClasses`) | ✅ | ✅ | ❔ | ❔ | 034, 052 |
| `bool_members` | `bool` members of the class and implicit conversions to `bool` | ✅ | ✅ | ❔ | ❔ | 079 |
| `cpp_constants` | `constexpr`, `static constexpr` locals, `INFINITY`, `std::nextafter`, C-style casts in constants | ✅ | ✅ | ❔ | ❔ | 079 |
| `default_args` | Default arguments of functions | ✅ | ✅ | ❔ | ❔ | 079 |
| `enum` | `enum` constants | ✅ | ✅ | ❔ | ❔ | 018, 019 |
| `switch` | `switch` statements | ✅ | ✅ | ❔ | ❔ | 046, 050 |
| `implicit_casts` | Implicit conversions between scalar types | ✅ | ✅ | ❔ | ❔ | 032 |
| `litemath` | LiteMath vectors and matrices (`float4`, `float4x4`, `dot`, ...) | ✅ | ✅ | ✅ | ✅ | 031, 043, 050 |
| `complex` | `complex` numbers from LiteMath | ✅ | ✅ | ❔ | ❔ | 028 |
| `std_math` | `std::min`, `std::max`, `std::abs`, `std::sqrt`, ... (as declared in `TINYSTL/`) | ✅ | ✅ | ✅ | ✅ | 061, 046 |
| `loop_bounds` | Arbitrary start and end expressions of the thread loops | ✅ | ✅ | ❔ | ❔ | 001, 002 |
| `loop_break` | `break` out of a loop over kernels in an RTV control function | ✅ | ✅ | ❔ | ❔ | 067, app 03 |
| `multi_loop` | Several consecutive thread loops in one IPV kernel | ✅ | ❌ | ❔ | ❔ | 063 |
| `kernel_locals` | Locals declared before the thread loop, used inside it | ✅ | ❌ | ❔ | ❔ | 062 |
| `threadlocal_array` | Local arrays, `[[threadlocal]]` for large ones | ✅ | ✅ | ❔ | ❔ | 033 |
| `vec_data_ptr` | Passing `m_vec.data()` of member vectors to kernels | ✅ | ✅ | ❔ | ❔ | 027 |
| `local_global_ptr` | Local pointers to buffers (`const float* p = a_in + off`) | ✅ | ✅ | ❔ | ❔ | 047, 077 |
| `nested_global_ptr` | Pointers to buffers passed through several function calls | ✅ | ✅ | ❔ | ❔ | 074, 076 |
| `global_ptr_copies` | Automatic copies of functions called with different buffers | ✅ | ✅ | ❔ | ❔ | 078 |

**Not supported in kernels** on any back end: recursion; calling kernels from kernels and control
functions from control functions; `new`/`malloc`, local `std::vector` and `std::string`;
`std::function` and lambdas with captures (a lambda is allowed only as the comparator of
`std::sort`); exceptions; `printf`, `std::cout` and file I/O; pointer members of the class; nested
containers (`std::vector` inside a struct stored in the class).

## Maintaining this page

- When you add a feature, add a row with a new ID and at least one test or sample.
- When a feature starts or stops working on some back end, change its mark in the same commit. If
  a test variant gets `"skip"` in `ktest.json`, the mark of that back end must not stay ✅.
- Keep IDs stable and short (`snake_case`).
- Keep the table format: kslicer parses it for `requireFeatures` (rows start with `` | `id` | ``,
  header rows start with `| ID |`, columns are named `Slang`, `GLSL`, `CUDA/HIP`, `ISPC`).
- Mark a feature ✅ for a back end only if a test runs it there. After `python3 ktest.py -b
  slang,glsl,ispc -s all` update the date of the last full check above.
