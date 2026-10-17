# AGENTS.md

Instructions for AI coding agents working in this repository. Human-oriented documentation is in
[README.md](README.md).

## What this repository is

kernel_slicer (`kslicer`) is a clang-based source-to-source translator (C++17, LLVM/clang 17). It
reads one C++ class written in a restricted style and generates a derived class with a GPU
implementation: Vulkan host code plus shaders (Slang by default, GLSL as legacy). There are also
CUDA/HIP, ISPC and WebGPU back ends.

There are two different kinds of tasks here. Find out which one you have before you start:

1. **Writing a program for kernel_slicer**: a new sample, porting an algorithm, a test input.
   Read [.claude/skills/kernel-slicer/SKILL.md](.claude/skills/kernel-slicer/SKILL.md) and its
   `references/` first. It is plain Markdown and is not specific to Claude: it gives the coding
   rules for input code, `main.cpp`/`CMakeLists.txt`/`kmake.json` templates and the verification
   procedure.
2. **Changing the translator itself**: bug fixes, new features, back ends, templates. The rest of
   this file is about this case.

## Build

```bash
cd cmake-build-release && make -j16     # existing build directory; produces ./kslicer
```

From scratch: `cmake -DCMAKE_BUILD_TYPE=Release ..` in `cmake-build-release/` (needs LLVM/clang 17
development packages; see README for other distributions and Windows). A debug build lives in
`cmake-build-debug/`.

**Rebuild kslicer after pulling or editing C++ sources.** The templates (`templates*/`) are read at
run time, so template edits need no rebuild. But a binary older than the templates may fail with
inja errors such as `variable 'X' not found`.

## Running the translator

Run from the repository root (templates are found relative to the current directory), or pass
`-selfdir <repo root>`. The command line for every test and sample is in
`.vscode/launch_slang.json` (Slang) and `.vscode/launch_glsl.json` (GLSL). Typical form:

```bash
./cmake-build-release/kslicer $PWD/apps/tests/023_prefix_summ/test_class.cpp -mainClass PrefSummTest \
  -stdlibfolder $PWD/TINYSTL -pattern ipv \
  -I$PWD/apps/LiteMath ignore -I$PWD/apps/LiteMathAux ignore -I$PWD/TINYSTL ignore \
  -shaderCC slang -DKERNEL_SLICER -v > kslicer.log 2>&1
grep -n "error:" kslicer.log
cd apps/tests/023_prefix_summ/shaders_generated && bash build_slang.sh   # GLSL: bash build.sh
```

Then build and run the sample: `mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make`,
and run `./build/testapp` (CPU) and `./build/testapp --gpu` from the sample folder, because shaders
are loaded relative to the working directory. Tests write `zout_cpu.json` / `zout_gpu.json` (or images);
compare them.

Pitfalls:
- **kslicer returns exit code 0 even on failure.** Check its log for `error:` and check that
  `build_slang.sh` compiled every shader.
- The Slang and GLSL back ends write into the same `shaders_generated/` folder. Before testing one
  back end, delete the folder so that stale `.spv` files of the other back end are not picked up.
- The sample apps need `apps/volk` built once (`cd apps/volk && cmake . && make`).

## Tests

- `apps/tests/NNN_name/`: one feature per test (hand-written `test_class.h/.cpp`, `main.cpp`,
  `CMakeLists.txt`). `apps/NN_name/`: larger samples.
- The test runner `python_scripts/tests.py` takes its list of tests from `.vscode/launch.json`.
  Copy `launch_slang.json` or `launch_glsl.json` over it, then
  `cd python_scripts && python3 gen_sample_list.py && bash run_tests.bat`. See
  `python_scripts/README.md`.
- When you add a test, add its launch configuration to `.vscode/launch_slang.json` and
  `.vscode/launch.json` (and `launch_glsl.json` if GLSL applies). Keep the numeric order and the
  style of neighbouring entries.
- Verify GPU changes on a real GPU, not only by building. Parallel primitives (scan, sort,
  reductions) must also be checked on large inputs (millions of elements, random sizes, many runs):
  races between workgroups often show up only there.
- After a change to shared code (host templates, `tr_host_vulkan.cpp`, rewriters), check both the
  Slang and the GLSL back ends.

## Code map

| Path | Role |
|------|------|
| `kslicer_main.cpp` | entry point, command line / `kmake.json` parsing, the sequence of passes |
| `initial_pass*.cpp`, `ast_matchers.cpp`, `extractor.cpp` | AST analysis: main class, kernels, control functions, data members |
| `class_gen*.cpp`, `class_data.cpp` | IPV/RTV/virtual-function specific processing, data layout |
| `rewrite_kernel*.cpp`, `rewrite_functions.cpp` | rewriting kernel and helper code into shader code |
| `reduction.cpp`, `hashmap.cpp`, `setter_process.cpp`, `block_expansion.cpp` | individual algorithmic features |
| `shaders_slang.cpp`, `shaders_glsl.cpp`, `shaders_cuda.cpp`, `shaders_ispc.cpp`, `shaders_clspv.cpp` | per back end: shader rewriting rules and shader file generation (`build_*.sh` script) |
| `tr_host_vulkan.cpp`, `tr_host_cuda.cpp`, `tr_host_ispc.cpp`, `tr_host_wgpu.cpp` | per back end: rewriting control functions into host code (including service calls: `memcpy`, scan, sort) |
| `template_rendering*.cpp` | building the JSON passed to the templates |
| `templates/` | Vulkan host code templates (`vk_class*.h/.cpp`), [inja](https://github.com/pantor/inja) syntax |
| `templates_slang/`, `templates_glsl/` | shader templates, including service shaders `z_scan_*`, `z_bitonic_*`, `z_memcpy` |
| `templates_cuda/`, `templates_ispc/`, `templates_wk/` | CUDA, ISPC, WebGPU templates |
| `TINYSTL/` | minimal standard library used when parsing input code |
| `doc/README_internal.md` | description of the translator passes (partly outdated) |

A feature that exists for one shader back end is often missing in another one. When you add or
fix something in `templates_glsl/` or `shaders_glsl.cpp`, check whether `templates_slang/` /
`shaders_slang.cpp` need the same change, and vice versa.

## Rules

- **Generated files are not committed.** `*_generated*`, `build/`, `cmake-build-*/` are ignored by
  `.gitignore`; do not force-add them. Commit the hand-written test sources, templates and
  translator sources.
- The submodules `apps/LiteMath`, `apps/vkutils`, `apps/volk`, `apps/LiteScene`, `apps/RayTracing`
  often have local changes. Do not commit submodule pointer changes unless asked.
- Commit only files related to your task: the working tree usually contains unrelated untracked
  output of other experiments.
- Slang is the main back end and GLSL is legacy, but both must keep working. The WebGPU back end
  (`tr_host_wgpu.cpp`, `templates_wk/`) is experimental: do not change it unless the task is about it.
- Match the surrounding code style: 2-space indentation, opening braces on their own line for
  functions and blocks, `if(` / `for(` without a space, `m_` prefix for members, `a_` prefix for
  function arguments, comment density similar to neighbouring code.
- In templates, keep generated code readable: it is meant to be read and extended by users.
