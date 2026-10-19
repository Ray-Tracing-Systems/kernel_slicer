# 29_conv2d: 2D image convolution (3x3, 5x5, 7x7)

Written with the `kernel-slicer` Claude skill (`.claude/skills/kernel-slicer`). IPV pattern only.

Main class `Convolution2D` (`conv2d.h`, `conv2d.cpp`):

| Control function | Kernels | What it shows |
|------------------|---------|---------------|
| `Convolve2D` | `kernel2D_Convolve` | direct K×K convolution, weights passed as a const input pointer, clamp-to-edge border |
| `ConvolveSeparable` | `kernel2D_ConvolveRow`, `kernel2D_ConvolveCol` | two passes through the member buffer `m_temp` |
| `ToLDR` | `kernel1D_FindMaxAbs`, `kernel2D_ToLDR` | parallel max-reduction into `m_maxValue`, then per-pixel tone mapping that reads it |

`main.cpp` applies a filter bank to `../images/nancy_church_2.hdr`: box 3×3, sharpen 3×3,
Laplacian edge 3×3, Gauss 5×5, emboss 5×5, Gauss 7×7 (2D) and Gauss 7×7 (separable).
It saves `zout_cpu_<filter>.bmp` and/or `zout_gpu_<filter>.bmp`.

## Build and run

Open this folder in VS Code: tasks build into `build_release/` and `build_debug/`, launch
configurations run the CPU, GPU and compare modes. From a terminal:

```bash
# translate (from this folder; -selfdir points kslicer to its templates in the repo root)
../../cmake-build-release/kslicer $PWD/kmake.json -selfdir $PWD/../..
cd shaders_generated && bash build_slang.sh && cd ..
cmake -S . -B build_release -DCMAKE_BUILD_TYPE=Release && cmake --build build_release -j 8

./build_release/testapp             # CPU
./build_release/testapp --gpu       # GPU (Vulkan)
./build_release/testapp --compare   # both, prints timings and max difference, exit code 1 on mismatch
```

`cmake -DUSE_VULKAN=OFF` builds the CPU-only version without the generated files.
