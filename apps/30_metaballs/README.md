# 30_metaballs: ray traced metaballs with reflections (IPV)

Written with the `kernel-slicer` Claude skill (`.claude/skills/kernel-slicer`). Uses only the IPV
pattern: every pass is a "parallel for" over pixels, like the image filters in `29_conv2d`.

Main class `Metaballs` (`metaballs.h`, `metaballs.cpp`), control function `Render(w, h, spp, out)`:

| Kernel | What it does |
|--------|--------------|
| `kernel2D_Trace` | per pixel: `spp × spp` supersampling; iterative path with up to 4 hits (3 mirror reflections); metaball surface found by uniform marching inside the ray's influence interval + bisection; hard shadows; checkerboard floor; sky |
| `kernel1D_AvgLogLum` | sum-reduction of log luminance into `m_sumLogLum` (auto exposure) |
| `kernel2D_ToneMap` | exposure from the reduction, Reinhard, gamma, RGBA8 output |

Metaballs field: `F(p) = Σ (1 - |p-c_i|²/R_i²)³` for `|p-c_i| < R_i`; surface `F(p) = 0.25`.
Scene data (centers/radii, colors/reflectivity) lives in member vectors set on the host.

## Build and run

Open this folder in VS Code: tasks build into `build_release/` and `build_debug/`, launch
configurations run the CPU, GPU and compare modes. From a terminal:

```bash
# translate (from this folder; -selfdir points kslicer to its templates in the repo root)
../../cmake-build-release/kslicer $PWD/kmake.json -selfdir $PWD/../..
cd shaders_generated && bash build_slang.sh && cd ..
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release && cmake --build build_release -j 8

./build_release/testapp                # CPU  -> zout_cpu.bmp (1024x1024)
./build_release/testapp --gpu          # GPU  -> zout_gpu.bmp
./build_release/testapp --compare      # both, prints timings and image difference
./build_release/testapp --gpu --spp 4  # 4x4 samples per pixel (default 2x2)
```

Measured on RTX 4090 / 2x2 spp: CPU (OpenMP) ≈ 740 ms, GPU ≈ 6.7 ms. With `--compare`
about 0.005% of pixels differ by more than 2/255, all on silhouettes: ray marching is sensitive
to the last float bit.
