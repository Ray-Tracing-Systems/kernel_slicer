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

## Measurements (old HydraCore3), done 2026-11-24

- GPU references: `profiling_results/ispc/make_references.sh` (PT 4096 spp, RTAO 1024 spp: more samples do not fit the
  Sobol AO tile; also 64 spp images for comparison with the same random numbers).
- Chunk sizes (`profiling_results/ispc/tune/results.txt`): wavefront 512 (16384 is 26-36% slower for PT), megakernel 256
  (almost no difference); written to `kmake_wave_ispc.json` / `kmake_mega_ispc.json`. Default of `-ispc_wave_chunk` is 1024.
- `testing/run_ispc_bench.py run` (CPU binaries `hydra_cpu` from `build_cpu` with `HYDRA_EXE_NAME=hydra_cpu`,
  `hydra_ispc_wave`, `hydra_ispc_mega`), 1024x1024, 64 spp, 3 runs; `report` writes `profiling/reports/ispc-mega-wave.tex`.
  The same file is `tables/ispc_mega_wave_cpu.tex` in the paper folder, section `cpumegawave` of `supplementary.tex`.
- Results, 256 spp (x4): megakernel is faster than wavefront by 10-16% (AO) and 8-35% (PT); ISPC is 1.9-2.1 (AO) and
  1.4-2.0 (PT) times faster than the input C++ code. PSNR against references 22.5-36 dB (kitchen PT 22.5 dB is Monte Carlo
  noise: GPU at 64 spp gives the same), against GPU with the same 64 spp 42-61 dB.

## Next steps

1. New HydraCore3 (`/home/frol/PROG/gml_private/HydraCore3`, root `kmake_mega.json` / `kmake_wave.json`): the same
   steps; expect more control functions and IPV/BE kernels.
2. Possible ISPC performance work: SoA payload in wavefront, fewer gathers of AoS scene data.

## Pitfalls found

- ISPC `float2/float3/int2/uint2/...` take 16 bytes (8/12 in C++): structures with such fields must not be stored in
  buffers.
- A global `static uniform const T * uniform g;` in ISPC is constant itself: assignment to it is lost silently.
- `print` from ISPC appears only if the executable is really relinked; touch the `.o` after `z_build_ispc.sh`.
- Code inside `#if` blocks excluded by `defines` is copied to ISPC as text, so the same macros are written to the ISPC file.
