# kslicer options and config files

kslicer takes its settings from the command line, from a JSON config file, or from both.

```bash
kslicer <input.cpp> [options]              # command line only
kslicer <config.json> [options]            # config file; command line options override it
kslicer -config <config.json> [options]    # the same
```

Every option has the form `-name value`. Exceptions: `-v` takes no value, and defines are written
as `-DNAME`. Boolean options take `0` or `1`.

kslicer loads its code templates relative to the current directory. Run it from the repository
root or pass `-selfdir <repository root>`.

## Command line options

### Input

| Option | Default | Meaning |
|--------|---------|---------|
| `<file>.cpp` | | Input source file. Several files may be given; they are merged into a temporary `<first>_temp.cpp`. Generated files are named after the first file |
| `-mainClass Name` | `TestClass` | Class to translate |
| `-stdlibfolder <dir>` | | Minimal standard library used for parsing; always `TINYSTL` from the repository |
| `-I<dir> ignore` | | Headers from `<dir>` are used for parsing only and are not translated into shaders (LiteMath, TINYSTL) |
| `-I<dir> process` | | Functions and types from headers in `<dir>` are translated into shader code. Headers next to the input file are always processed |
| `-ignore <file>` / `-process <file>` | | The same for a single header, as an exception to the folder rules |
| `-DNAME` | | Preprocessor define for parsing; also written to the merged temporary file. `KERNEL_SLICER` is always defined |
| `-cpp_include <file>` | | Extra `#include` line in the generated header |
| `-selfdir <dir>` | current dir | Repository root, used to find templates |
| `-v` | | Verbose output |

### Back end and output

| Option | Default | Meaning |
|--------|---------|---------|
| `-shaderCC <name>` | `clspv` | Back end: `slang` (main, Vulkan + Slang), `glsl` (legacy, Vulkan + GLSL), `cuda`, `hip`, `musa`, `cuda_slang`, `ispc`, `wgpu`, `clspv` (outdated). **Always set it** |
| `-suffix _Name` | `_Generated` | Suffix of the generated class (`MyClass_Generated`), of the factory function (`CreateMyClass_Generated`) and of the shader folder (`shaders_generated`) |
| `-shaderFolderPrefix <path>` | | Prefix of the shader path used by the generated host code to load `.spv` files |
| `-gen_gpu_api 1` | `0` | Additionally generate an interface `I<MainClass><Suffix>` with the GPU-specific API (`...Cmd` functions, buffer setters) that the generated class implements |
| `-timestamps 1` | `0` | GPU timestamps for every kernel; read them with `GetExecutionTime("kernel2D_Name", t)`: `t[0]` average, `t[1]` min, `t[2]` max in ms |
| `-pipelinecache 1` | `0` | Use a Vulkan pipeline cache when creating pipelines |
| `-mt_shader_compile 0` | `1` | Generate a sequential shader build script instead of the parallel one (Linux; Windows `.bat` scripts are always sequential) |
| `-skip_shader_includes 0` | `1` | Add `-I..` and include folders to the `slangc` command lines (usually not needed) |
| `-auxshadercc "<flags>"` | | Extra flags for the shader compiler in the build script |
| `-spirv_ver <ver>` (alias `-cc`) | compiler default | Target SPIR-V version, e.g. `13` for SPIR-V 1.3 |
| `-vulkan_ver <ver>` | `12` | Target Vulkan version; `11` limits SPIR-V to 1.3 |

### Kernels and execution

| Option | Default | Meaning |
|--------|---------|---------|
| `-reorderLoops YX` | `XY` | For 2D IPV kernels written with `y` outer and `x` inner: makes `x` the fastest GPU thread index. **Always use it for images**: memory-bound kernels become several times faster |
| `-megakernel 1` | `0` | Each RTV control function becomes one shader instead of a sequence of kernels |
| `-persistent 1` | `0` | Persistent threads for RTV control functions |
| `-enableSubgroup 0\|1` | not set | Allow subgroup operations (reductions, hash tables). If not set, only optimizations that are always correct are used where the device supports them |
| `-warpSize <n>` | `32` | Subgroup size assumed by reductions to skip synchronization |
| `-atomicf_emul 1` | `0` | Emulate float atomics (`InterlockedAdd` on float, `InterlockedAdd3f`, `ReduceAdd<float>`) with a CAS loop on a `uint` view of the buffer, for GPUs without `shaderBufferFloat32AtomicAdd`. Slang only; `2` is the same as `1` |
| `-pattern ipv\|rtv` | | Ignored: the pattern is chosen by the kernel name prefix. Kept for old launch configs |

### Ray tracing

| Option | Default | Meaning |
|--------|---------|---------|
| `-enable_ray_tracing_pipeline 1` | `0` | With `-megakernel 1`: use the Vulkan ray tracing pipeline (raygen and hit shaders) instead of ray queries in compute shaders, if the class uses acceleration structures |
| `-force_ray_tracing_pipeline 1` | `0` | The same, and always request ray tracing features |
| `-enable_callable_shaders 1` | `0` | Use callable shaders in the ray tracing pipeline |
| `-enable_motion_blur 1` | `0` | Motion blur support; implies the ray tracing pipeline |
| `-bothrt 1` | `0` | Wrap the scene member into a proxy that keeps both the user (software) implementation and the Vulkan hardware one, even without custom intersection shaders |

### Data layout and features

| Option | Default | Meaning |
|--------|---------|---------|
| `-const_ubo 1` | `0` | Kernels never write class members: the uniform data buffer is declared read-only in shaders |
| `-uniform_ubo 1` | `0` | Use a Vulkan uniform buffer instead of a storage buffer for class data |
| `-sortubo 1` | `0` | Sort scalar class members in the uniform data structure by size |
| `-halfTex 1` | `0` | Use 16-bit float formats for float textures |
| `-forceEnableHalf`, `-forceEnableInt8`, `-forceEnableInt16`, `-forceEnableInt64` | auto | Force requesting the corresponding shader arithmetic type features |
| `-force8bitStorage`, `-force16bitStorage` | auto | Force requesting 8/16-bit storage features (they are detected automatically from member types) |

### Back end specific

| Option | Default | Meaning |
|--------|---------|---------|
| `-useCUB 0` | `1` | CUDA: do not use CUB for scan and sort |
| `-skip_ubo_read 1` | `0` | CUDA: do not read class data back from the GPU after control functions |
| `-ispc_threads <0\|1\|2>` | `0` | ISPC threading model: `0` ISPC tasks (`launch`, needs `ispc_tasksys.cpp`), `1` single thread, `2` OpenMP loop over blocks, each block is an ISPC call |
| `-ispc_target <target>` | `avx2-i32x8` | ISPC `--target` written to `z_build_ispc.sh`, e.g. `avx512skx-x16` |
| `-ispc_explicit_id 1` | `0` | ISPC: explicit thread indices |
| `-wgpu_ver <n>` | `30` | WebGPU: API version of the generated code |

Options not listed here (`-vec_in_ubo`, `-const_shit`, `-new_rawname`, `-fwdfundecl`,
`-makerInterfaceName`, `-composInterface`, `-composImplementation`, `-temp_suffix`, `-cl-std`)
are internal or used by specific projects.

## Config file (kmake.json)

A config file keeps the translator settings next to the sample and is easier to maintain than a
long command line. Paths in it are relative to the config file. Comments (`//`) are allowed.

```jsonc
{
  "mainClass"     : "MyAlgo",                  // the class to translate (-mainClass)
  "source"        : ["my_algo.cpp"],           // input files, merged in this order
  "baseClasses"   : [],                        // base classes of mainClass, from leaf to root
  "composClasses" : [],                        // composition, see below
  "kernels" : {                                // per-kernel settings
    "kernel2D_Trace" : {"wgSize": [8, 8, 1]}
  },
  "options" : {                                // any command line options, without the input file
    "-shaderCC"     : "slang",
    "-reorderLoops" : "YX",
    "-timestamps"   : "1"
  },
  "defines"        : {"MY_DEFINE": "1"},       // like -DMY_DEFINE=1
  "includeProcess" : [],                       // like -I<dir> process
  "includeIgnore"  : ["../LiteMath"],          // like -I<dir> ignore
  "end" : ""
}
```

Run it as `kslicer apps/my_app/kmake.json`, or from the sample folder as
`../../cmake-build-release/kslicer $PWD/kmake.json -selfdir $PWD/../..`. Options given on the
command line override the ones in `"options"`. A sample may have several configs for different
variants, e.g. `apps/tests/042_atomic_add_float/kmake_hw.json` and `kmake_emul.json`.

### Keys

| Key | Meaning |
|-----|---------|
| `mainClass` | Name of the main class |
| `source` | Array of input `.cpp` files |
| `baseDirectory` | Base folder for relative paths instead of the config folder |
| `baseClasses` | Base classes of the main class whose members and kernels are translated too, listed from leaf to root. Example: `apps/tests/034_inheritance_v1` |
| `composClasses` | Classes that the main class contains by pointer to an interface (composition). See below |
| `kernels` | Per-kernel settings by kernel name: `wgSize` (work group size). Default sizes for all kernels of one dimension: `default1D`, `default2D`, `default3D` |
| `options` | Command line options as `"-name": "value"` |
| `defines` | Preprocessor defines, `{"NAME": "value"}` |
| `includeProcess`, `includeIgnore` | Folders, as `-I<dir> process` and `-I<dir> ignore` |
| `excludedControlFunctions` | Names of public methods that must not be treated as control functions |
| `additionaIncludes` | Extra `#include` lines for the generated header, like `-cpp_include` (the key is spelled this way) |

Work group sizes: the defaults (256 for 1D kernels, 32×8 for 2D) are good in most cases. Set
`wgSize` for named kernels or with `default1D`/`default2D`/`default3D`. The keys `default` and
`all` apply one size to **every** kernel regardless of its dimension, and they also override
named entries. A 2D size under `default` breaks 1D reductions, so do not use it.

### Composition and custom intersection

`composClasses` describes a member of the main class that holds an interface implemented by
another class, typically the ray tracing scene. Only one composition is currently supported.

```jsonc
"composClasses" : [{
  "interface"      : "ISceneObject",          // interface type of the member
  "implementation" : "BFRayTrace",            // class that implements it and is translated too
  "intersection"   : {                        // optional: custom primitive intersection
    "shader" : "IntersectionShader",          //   intersection function of the implementation
    "accel"  : "m_pRayTraceImpl"              //   member that holds the acceleration structure
  }
}]
```

The intersection function is used both in hardware ray tracing (as an intersection shader) and in
software BVH traversal. Examples: `apps/tests/035_intersection_shader`,
`apps/tests/046_intersection_shader_v2` (`kmake_rq.json` for ray queries, `kmake_cs.json` for
compute shaders).
