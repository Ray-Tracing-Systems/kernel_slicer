# ISPC back end: stability — plan of the next stage

Status of 2026-11-24 (kslicer `15cd84bd`, branch `ispc_megakernel`). The previous stage (RTV megakernel/wavefront,
old HydraCore3, graphics pipeline) is described in [ispc_rtv_plan.md](ispc_rtv_plan.md).

## Goal

Most features are tested on ISPC only through a few samples: in [features.md](features.md) the ISPC column is mostly
❔. The goal is to run the regular tests on ISPC, fix the translator where they fail, and fill the ISPC column with
results. After that: the new HydraCore3 on ISPC (item 1 of "Next steps" in `ispc_rtv_plan.md`).

## Triage (2026-11-24)

Of 142 tests and apps, 44 already have an ISPC variant (OK, except 085 and 086: `push_back` in kernels). The other 98
were run once with a temporary `ispc` variant (first `kmake` of the folder, no options) up to the build stage:

- **27 generate and compile with ISPC**: 021, 023, 024, 025, 026, 027, 034, 041, 042, 043, 044, 047, 048, 049, 052,
  057, 059, 060, 068, 088, 089, 090, 091, app 19, app 28, app 29, app 30. They only need the host part of the test
  (see step 1); results are not checked yet.
- **9 fail in the host code**: the ISPC class marks kernels `override`, but IPV kernels of these tests are not
  `virtual`: 001, 002, 009, 010, 011, 012, 015, 016, 017.
- **62 fail in the ISPC compiler.** Root causes, by the number of tests:

| # | Cause (first errors) | Tests |
|---|----------------------|-------|
| C1 | Names of the generated wrapper are missing in kernels with textures/images, buffer reference, hash map, half: `a_ubo`, `_start/_end/_span`, `a_width/a_height`, `a_out`, `a_size`, `kernel*_Task` | 006, 008, 020, 039, 053, 055, 056, 065, 066, 073, app 14, app 15 |
| C2 | C math functions are not declared for ISPC: `fmax`, `fmin`, `fabs`, `copysign` | app 01, app 02, app 03, app 10, app 11, app 13, app 16, app 17, app 20, app 26, 065 |
| C3 | Composed classes and software ray tracing: `m_pRayTraceImpl_*` members, `this`, `kernel_RayTrace_Thread`, `kgen_*` of RTV inside composed class | 018, 019, 035, 036, 037, 038, 046, app 26 |
| C4 | Reductions in IPV kernels emit GPU code (`get_local_id`, shared arrays) | 003, 045, app 09 |
| C5 | Atomics on counters (`atomic_inc`) and `push_back` in kernels | 004, 005, app 08 (also 085, 086) |
| C6 | Structs and C++ constructs: forward declared / incomplete structs, constructors, default member initializers, member functions of structs, `float4x4` indexing `m[i][j]`, structs by value in `export` functions (`complex`), local arrays, storage class specifiers, `const` locals without initializer | 020, 022, 028, 031, 032, 033, 040, 061, 063, 064, 070, 075, 079, app 01, app 21 |
| C7 | Pointers to buffers through functions: missing `return`, no matching overload of copies of functions | 058, 062, 074, 076, 077, 078 |
| C8 | Virtual functions: pointer arithmetic on undefined struct, color fields (`red/green/blue/alpha`) | 069, app 10, app 13, app 26, app 27 |
| C9 | `half` / fp16: `f16tof32`, members of half vector types | 022, 053, 070 |
| C10 | LiteNN: labels and `:` in generated code (`Multiple labels named ...`) | 071, 072, 073 |

To repeat the triage: add `{"name": "ispc_triage", "backend": "ispc", "kmake": <first kmake>}` to every `ktest.json`
without an ISPC variant, run `python3 ktest.py -b ispc -s all --variant ispc_triage --stage build --no-skip`, read
`ktest_logs/<date>/summary.txt` and the logs (ISPC errors are lines `Error: ...`), then `git checkout` the `ktest.json`.

## Progress

- 2026-11-25: step 1 done for 26 of the 27 tests (app 19 is left: it renders CPU and GPU in one run with hand-written GPU
  code) and for the 9 tests with the `override` problem. The `override` problem is fixed: only `virtual` kernels are
  overridden, a control function which calls a kernel that is not virtual is copied to the ISPC class (as for block
  expansion kernels). Structs with 2-/3-component vectors in member vectors and POD members are converted to ISPC layout
  (021); such buffers passed by pointer from control functions are reported as an error (025 is skipped). ISPC variants:
  001 skipped (reduction, C4); 002 not converted yet (one run computes CPU and GPU, reduction).
- 2026-11-25: new HydraCore3 (step 5) works with ISPC in both modes without changes of the renderer: branch
  `ispc_hydra` of `/home/frol/PROG/gml_private` (commit 83ed463a9) adds `kmake_{mega,wave}_ispc.json`, `ispc_build.sh`,
  `USE_ISPC` in CMake and `--ispc` in `main.cpp`. Translator fixes it needed (kslicer 29440b76): layout conversion found
  from clang types of members (also through typedefs, `RandomGen`), split of `float a = x, b = a*y;`, `int4(float4)`
  casts, renaming of locals named as ISPC keywords (`half`), forward declarations of copies of functions with buffers,
  `memcpy` with `void*` (GCC `-Werror=class-memaccess`). Benchmark scenes (coffee, spaceship, staircase, kitchen),
  512x512, 16 spp, MIS-PT: ISPC images match C++ with 46-63 dB (megakernel and wavefront are identical), megakernel is
  1.2-1.5 times faster than the C++ code, wavefront 1.0-1.4 times (not tuned; old HydraCore3 gave 1.4-2.0 for PT).


1. **Host part of tests** (mechanical, no translator changes): for the 27 tests above add `USE_ISPC` to
   `CMakeLists.txt` (`*_ispc.cpp`, `*_kernels.o`, `../../LiteMathAux/ispc_tasksys.cpp`), `--ispc` to `main.cpp`
   (`Create*_ISPC()`, output `zout_ispc*`) and an `ispc` variant to `ktest.json`, as in `apps/tests/051_saxpy_ipv_rtv`.
   Run them; failures of comparison are new items of step 2.
2. **Translator fixes**, in the order of the table (each fix enables the ISPC variant of its tests):
   - first the host `override` problem (only kernels which are `virtual` may be overridden, otherwise override the
     control function) and C2 (declare the functions in `LiteMathISPC.h`) — cheap and unblock many tests;
   - C1 (one place in the ISPC kernel templates, most likely the branch of kernels with textures/UBO access in
     functions), C5, C4 — they are needed by HydraCore3 and by service functions;
   - C3 and C8 — needed for software ray tracing and virtual functions (new HydraCore3 uses virtual materials less,
     but uses composition);
   - C6, C7, C9, C10 — by demand.
3. **New ISPC-specific tests** (`apps/tests/1NN_ispc_*`, ISPC compared with C++, bit-exact where the code is
   deterministic):
   - layout: `float2/float3/int2/uint2` fields in structs stored in buffers and in member vectors (16 bytes in ISPC):
     either correct padding by kslicer or an error message instead of wrong results;
   - RTV megakernel/wavefront: thread count not multiple of the chunk, `-ispc_wave_chunk`/`-ispc_mega_chunk` variants,
     payload structs with arrays, `break` and early exit of some threads;
   - textures: RGBA8/sRGB/float/float4 in one array, wrap/clamp, texture writes (`Image2D` written by kernels);
   - rasterization: blending (the linear 4x4 path must be off), RGBA8 and float4 targets, `LINEAR` layout, partial
     blocks at screen edges, depth-only pass — pixel-exact with C++;
   - parallel primitives (scan, sort, reductions, atomics) on large random sizes and many runs (races between OpenMP
     threads and ISPC tasks).
4. **features.md**: replace ❔ of the ISPC column by the results of steps 1–3 in the same commits.
5. **New HydraCore3** on ISPC (`/home/frol/PROG/gml_private/HydraCore3`, `kmake_mega.json` / `kmake_wave.json`).

## Notes

- `OMP_NUM_THREADS` does not limit ISPC tasks (`launch[]`, `ispc_tasksys.cpp`), only OpenMP loops of the host code.
- On AVX2 ISPC has no scatter and no vector `int64 -> float`: write contiguous pixels/elements with `programIndex`
  indexing from a `uniform` base, convert int64 with `RS_Int64ToFloat` (`raster_ispc.h`).
- Generated files of a test are shared by back ends: `ktest.py` regenerates them for every variant.
