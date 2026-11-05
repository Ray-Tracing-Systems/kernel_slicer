# Writing code for kernel_slicer

The translator supports a subset of C++. In brief:

- Every pointer argument of a control function has a size annotation `[[size("expr")]]`. `const`
  pointers are inputs and non-`const` pointers are outputs.
- All intermediate buffers are `std::vector` fields of the class, sized before `CommitDeviceData()`.
- Kernels must not call kernels, and control functions must not call control functions.
- Kernels must not allocate memory, do I/O, throw exceptions, use recursion or pointer fields.
- Use the vector types and functions of [LiteMath](https://github.com/msu-graphics-group/LiteMath)
  (`float4`, `float4x4`, `dot`, `normalize`, ...).
- Debug on the CPU first, then translate and compare CPU and GPU results.

The rules, templates of `main.cpp`/`CMakeLists.txt`/`kmake.json`, common mistakes and a map of
samples are in [`.claude/skills/kernel-slicer/SKILL.md`](../.claude/skills/kernel-slicer/SKILL.md).
This file is written for LLM agents but reads as an ordinary guide.

Which features and C++ constructs are supported by each back end, and which tests show them, is
listed in [features.md](features.md).

## Older tutorials

These are partly outdated (they use the clspv back end):
[bloom filter (IPV)](tutorials/ex05_bloom_ipv.md),
[single kernel (RTV)](tutorials/ex00_single_kernel.md),
[passing data between kernels](tutorials/ex01_passing_data.md),
[path tracing with loop break](tutorials/ex03_path_tracing_loop_break.md),
[glue code and overriding generated code](tutorials/glue_code.md).

## AI assistants

The input language is ordinary C++, so the approach works well with coding assistants.
The repository contains:

- [`.claude/skills/kernel-slicer/`](../.claude/skills/kernel-slicer/): a skill for LLM agents that
  write programs for kernel_slicer, covering both patterns, the host code, building and verification.
  Claude Code loads it automatically when working in this repository. The `apps/30_metaballs` and
  `apps/ai_generated` samples were written with it;
- [`AGENTS.md`](../AGENTS.md): instructions for agents that work on the translator itself (building,
  testing, code structure).
