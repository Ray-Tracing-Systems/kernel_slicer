# ISPC: RTV megakernel and wavefront, HydraCore3 on CPU — status and plan

Branch `ispc_megakernel` of kernel_slicer, branch `occupancy_test` of the old HydraCore3
(`/home/frol/PROG/HydraRepos/HydraCore3`).

## Goal

The ISPC back end supports the RTV pattern with `-megakernel 1` and `-megakernel 0` (persistent rays are not needed).
HydraCore3 is translated to ISPC in both modes, ray tracing performance of megakernel and wavefront is measured on
CPU for the paper (a table like Table 5, `tab:rtao-mega` in `kslicer_ieee_tvcg/crossrt_ieee_template.tex`).

Agreed conditions:
- Hydra may be simplified (complex materials can be dropped, hard translator problems can be worked around in Hydra).
- Wavefront on CPU processes threads in chunks (cache locality).
- Threading: OpenMP.
- Measurements: 4 scenes of the paper (coffee, spaceship, staircase, kitchen from `scenes/benchmark`), 1024x1024,
  64 spp (x4 gives 256 spp of the paper), AO and PT, AVX2 (`avx2-i32x8`). Result: a separate LaTeX file in
  Hydra `profiling/reports`.
- Correctness: PSNR 25-30 dB against GPU references with 4096 spp.

## Done

kernel_slicer (`git log master..ispc_megakernel`):
- `d047da26` RTV for ISPC: megakernel (code of a thread is ISPC function, OpenMP over chunks of `-ispc_mega_chunk`
  threads) and wavefront (control function on the host for chunks of `-ispc_wave_chunk` threads, every kernel
  processes all threads of the chunk, thread flags as in Vulkan, payload arrays of the chunk for each OpenMP
  thread). Pointer parameters which do not point to buffers are ISPC references; member vectors of
  `float2/float3/int2/...` are passed as padded copies (ISPC short vectors take 16 bytes); `-ispc_process_include 1`.
  Tests: 051, 067 (also small chunks), app 07 (path tracing with Embree).
- `da4ba4c8` arrays of textures `std::vector<std::shared_ptr<ICombinedImageSampler>>` (test 007).
- `efbd1fc4` fixes found on Hydra: wavefront payload is loaded/stored by the kernel and passed to code of a thread by
  reference; `defines` of config go to the ISPC file; `auto`, `float3 v(a,b,c)`, `complex{}`; LiteMathISPC.h:
  complex, component-wise functions, `inverse3x3`, ...
- `854d5a00` sRGB textures.
- `doc/features.md` (`rtv`, `megakernel`, `loop_break`, textures for ISPC) and `doc/options.md` are updated.
- ISPC regression: all ISPC tests OK (007 has WARN because of FMA in its Mandelbrot loop, GPU has it too);
  all Slang tests OK.

HydraCore3 (`b6b16ec` on `occupancy_test`):
- `bash ispc_build.sh /home/frol/PROG/kernel_slicer [mega|wave]` generates the ISPC implementation (configs
  `kmake_mega_ispc.json`, `kmake_wave_ispc.json`) and builds `bin-release/hydra_ispc_mega`, `bin-release/hydra_ispc_wave`
  (run with `--ispc`); CMake options `USE_ISPC`, `ISPC_DIR`, `HYDRA_EXE_NAME`.
- Changes of Hydra: `EmbreeRT.h`; `RandomGen` and `LightSource` without `uint2/float2` (they are in buffers);
  `DISABLE_FILM`; ISPC configs also define `DISABLE_LENS` and `DISABLE_SPECTRUM` (benchmark scenes are RGB, pinhole).
- Checked: mega and wave images are bitwise equal; against C++ CPU at 64 spp, 256x256: PSNR 50-57 dB (coffee,
  spaceship), 52 dB (staircase), 42 dB (kitchen); AO is equal too.
- First timings, 256x256, 4 spp, PT: mega is 1.9-3.6 times faster than wave (wave chunk 4096, mega chunk 256 are not tuned).
- GPU references (hw-mega build of the same sources, `profiling_results/ispc/gpu_hw_mega/`):
  `profiling_results/ispc/make_references.sh` renders 1024x1024, 4096 spp, mispt and rtao into
  `profiling_results/ispc/references/` (log: `profiling_results/ispc/make_references.log`).

## Next steps

1. Check that the 4 PT references are rendered (`rc=0` in the log). RTAO references failed on GPU: 4096 spp is too
   many for the Sobol AO tile ("Sobol AO tile did not contain the requested samples for every pixel"); render them
   with fewer spp (for example 256 or 1024, AO converges fast) by changing the script for `rtao`.
2. Tune chunk sizes on one or two scenes: `-ispc_wave_chunk` 1024 / 4096 / 16384, `-ispc_mega_chunk` 64 / 256 / 1024
   (set in `kmake_*_ispc.json` or on kslicer command line), keep the best for each mode and write them to the report.
3. Measurement script in Hydra (for example `testing/run_ispc_bench.py`): 4 scenes x {mispt, rtao} x {mega, wave},
   1024x1024, 64 spp, 3 runs, median of `PathTraceBlock(exec)` / `AmbientOcclusionBlock(exec)`; PSNR of every image
   against the reference (expected 25-30 dB and more). Optionally C++ OpenMP column: its `GetExecutionTime` looks
   wrong (always ~250 ms for PT at 256x256), measure wall time instead.
4. LaTeX table in `profiling/reports/ispc-mega-wave.tex` (like Table 5: AO and PT, wave/mega, CPU i9-10940X,
   14 cores, AVX2), commit to `occupancy_test` (without images and binaries).
5. Documentation in kernel_slicer: `doc/concepts.md` / features about RTV on ISPC (chunks, references, payload
   load/store), maybe test 067 variant with `-ispc_threads`; commit.
6. New HydraCore3 (`/home/frol/PROG/gml_private/HydraCore3`, root `kmake_mega.json` / `kmake_wave.json`): the same
   steps; expect more control functions and IPV/BE kernels.

## Pitfalls found

- ISPC `float2/float3/int2/uint2/...` take 16 bytes (8/12 in C++): structures with such fields must not be stored in
  buffers.
- A global `static uniform const T * uniform g;` in ISPC is constant itself: assignment to it is lost silently.
- `print` from ISPC appears only if the executable is really relinked; touch the `.o` after `z_build_ispc.sh`.
- Code inside `#if` blocks excluded by `defines` is copied to ISPC as text, so the same macros are written to the ISPC file.
