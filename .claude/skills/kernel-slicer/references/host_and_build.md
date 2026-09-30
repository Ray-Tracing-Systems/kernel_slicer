# Host code, CMake, kmake.json and running kslicer

Put a new application in `apps/<NN_name>/` so that the relative paths below work
(for `apps/tests/<name>/` add one more `../`).

Hand-written files:

```
apps/NN_name/
  my_algo.h          main class declaration
  my_algo.cpp        kernels + control functions + helpers   <- file given to kslicer
  main.cpp           host program: CPU / GPU switch, I/O, comparison
  CMakeLists.txt
  kmake.json         optional translator config
```

kslicer writes next to them: `my_algo_generated.h`, `my_algo_generated.cpp`,
`my_algo_generated_ds.cpp`, `my_algo_generated_init.cpp`, and `shaders_generated/`
(the suffix is `_generated`, lower-cased from `-suffix`, default `_Generated`).

## main.cpp template (IPV)

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>

#include "my_algo.h"
#include "ArgParser.h"                 // apps/utils
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"                 // apps/utils, handy for comparing CPU/GPU numbers

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<MyAlgo> CreateMyAlgo_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures MyAlgo_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool onGPU = args.hasOption("--gpu");

  const int N = 1024*1024;
  std::vector<float> input(N), output(N);
  for(int i = 0; i < N; i++) input[i] = float(i % 100);

  std::shared_ptr<MyAlgo> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int deviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = MyAlgo_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, deviceId);
    pImpl         = CreateMyAlgo_Generated(ctx, N);      // N = max threads of any kernel
  }
  else
  #endif
    pImpl = std::make_shared<MyAlgo>();

  pImpl->Reserve(N);             // 1) size/reserve member vectors, set parameters
  pImpl->CommitDeviceData();     // 2) upload class data (call again after changing members)
  pImpl->Process(input.data(), output.data(), N);   // 3) control function(s)

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Process", timings);
  std::cout << "Process(exec) = " << timings[0] << " ms" << std::endl;
  std::cout << "Process(copy) = " << timings[1] + timings[2] << " ms" << std::endl;

  JSONLog::write("output", output);
  JSONLog::saveToFile(onGPU ? "zout_gpu.json" : "zout_cpu.json");

  pImpl = nullptr;               // destroy generated object before the context
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
```

Notes:
- For a `--compare` mode (CPU and GPU objects in one process, per-case max difference, exit
  code 1 on mismatch) copy the structure of `apps/29_conv2d/main.cpp`.
- The factory name is `Create<MainClass><Suffix>`; with `"-suffix": "_GPU"` it becomes
  `CreateMyAlgo_GPU` and `MyAlgo_GPU_ListRequiredDeviceFeatures`.
- Constructor arguments of the main class are prepended to the factory arguments.
- `GetExecutionTime` on the generated class fills `[exec, copy-in, copy-out, overhead]`.
- For images use `LiteImage::SaveBMP(name, uint32Pixels, w, h)` from `apps/LiteMath/Image2d.h`
  and add `../LiteMath/Image2d.cpp` to the sources. Row 0 of the pixel array is the **bottom**
  row of the BMP: a renderer must map `y = 0` to the bottom of the screen.

## CMakeLists.txt template

```cmake
cmake_minimum_required (VERSION 3.8)
project (test)
set (CMAKE_CXX_STANDARD 17)
option(USE_VULKAN "Enable Vulkan implementation" ON)

include_directories(".")
find_package(OpenMP)
if(USE_VULKAN)
  find_package(Vulkan)
  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)
  include_directories(${Vulkan_INCLUDE_DIR})
  include_directories("../vkutils" "../volk")
  link_directories("../volk")
endif()
include_directories("../LiteMath" "../utils")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -fPIC -Wno-attributes")

set(MAIN_SOURCE main.cpp my_algo.cpp)

set(VKGEN_SOURCE my_algo_generated.cpp
                 my_algo_generated_ds.cpp
                 my_algo_generated_init.cpp)

set(VKUTILS_SOURCE ../vkutils/vk_utils.cpp
                   ../vkutils/vk_copy.cpp
                   ../vkutils/vk_buffers.cpp
                   ../vkutils/vk_images.cpp
                   ../vkutils/vk_context.cpp
                   ../vkutils/vk_alloc_simple.cpp
                   ../vkutils/vk_pipeline.cpp
                   ../vkutils/vk_descriptor_sets.cpp)

set(LIBS OpenMP::OpenMP_CXX)
if(USE_VULKAN)
  list(APPEND MAIN_SOURCE ${VKGEN_SOURCE} ${VKUTILS_SOURCE})
  list(APPEND LIBS ${Vulkan_LIBRARY} volk dl)
endif()

add_executable(testapp ${MAIN_SOURCE})
target_link_libraries(testapp LINK_PUBLIC ${LIBS})
```

`-Wno-attributes` silences GCC warnings about `[[size(...)]]`. Build the CPU-only version first
with `cmake -DUSE_VULKAN=OFF ..`; it must compile and give correct results before translation.

## Running kslicer

The binary is `cmake-build-release/kslicer` (or `cmake-build-debug/kslicer`) in the repository
root. It loads its code templates (`templates_slang/`, ...) relative to the current directory, so
run it from the repository root. From any other directory add `-selfdir <repo root>`; without it
kslicer aborts at step (7) with `basic_ios::clear: iostream error`:

```bash
cd apps/NN_name
../../cmake-build-release/kslicer $PWD/kmake.json -selfdir $PWD/../..
```

Command line form:

```bash
./cmake-build-release/kslicer apps/NN_name/my_algo.cpp \
    -mainClass MyAlgo \
    -stdlibfolder TINYSTL \
    -Iapps/LiteMath ignore -Iapps/LiteMathAux ignore -ITINYSTL ignore \
    -shaderCC slang -reorderLoops YX \
    -DKERNEL_SLICER -v
```

- `-I<dir> ignore`: headers from this folder are used for parsing only, not translated into
  shaders (LiteMath is built into the shader library already).
- `-I<dir> process`: helper headers from this folder are translated into shader code. Headers
  in the application folder itself are always processed.
- `-shaderCC slang | glsl | cuda | ispc | wgpu`. Use `slang` (current Vulkan back end); `glsl` is
  legacy. Always set it: the built-in default is the older clspv.
- `-megakernel 1`: RTV control functions become a single shader each.
- `-reorderLoops YX`: **always set it for 2D IPV kernels** written with `y` outer and `x` inner.
  It makes `x` the fastest GPU thread index, so neighbouring threads read neighbouring pixels.
  Memory-bound 2D kernels become 2–5 times faster (see the table in SKILL.md).
- `-suffix _GPU`: name of generated class and files.
- `-timestamps 1`: GPU timestamps per kernel; read them with `GetExecutionTime("kernel2D_Name", t)`
  (`t[0]` avg, `t[1]` min, `t[2]` max in ms). With `-megakernel 1` the merged RTV kernels are
  reported as `"<ControlFunction>Mega"`. Timestamps work for every control function, for IPV and
  RTV, with and without megakernel (fixed in kslicer: older translator builds recorded only the
  first control function and crashed when a megakernel class also had IPV control functions).
- `-pattern ipv|rtv` appears in old launch configs; the pattern is actually chosen by the
  kernel name prefix.

Config-file form, `kslicer apps/NN_name/kmake.json` (paths relative to the json file):

```jsonc
{
  "mainClass"     : "MyAlgo",
  "baseClasses"   : [],
  "composClasses" : {},
  "kernels" : {
    "kernel2D_BlurX" : {"wgSize": [16, 16, 1]}           // per-kernel options
  },
  "options" : {
    "-shaderCC"     : "slang",
    "-reorderLoops" : "YX",       // always, for 2D IPV kernels with y outer / x inner
    "-timestamps"   : "1"         // per-kernel GPU times in GetExecutionTime("kernel2D_...")
  },
  "source"         : ["my_algo.cpp"],
  "includeProcess" : [],
  "includeIgnore"  : ["../LiteMath"],
  "end" : ""
}
```

Work-group sizes: the defaults (256 for 1D kernels, 32×8 for 2D) are fine. Set `wgSize` only
per kernel name. **Do not put a 2D `wgSize` under `"default"`** when the class has 1D kernels:
it is applied to them too, and a 1D reduction kernel then returns wrong results on the GPU
(verified in `apps/29_conv2d` with both Slang and GLSL back ends).

Examples of complete command lines for every sample are in `.vscode/launch_slang.json`
(and in the legacy `.vscode/launch_glsl.json`).

## VS Code configuration

Copy `.vscode/tasks.json` and `.vscode/launch.json` from `apps/30_metaballs`: tasks translate
(with `-selfdir`), compile Slang shaders and build into `build_release/` and `build_debug/`;
launch configurations run CPU, GPU and `--compare` with the matching build as `preLaunchTask`.
The sample folder itself is opened as the workspace.

## Compiling shaders and running

```bash
cd apps/NN_name/shaders_generated
bash build_slang.sh      # Slang -> SPIR-V via slangc (-shaderCC slang)
# bash build.sh          # legacy: GLSL -> SPIR-V via glslangValidator (-shaderCC glsl)
cd .. && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j8
cd .. && ./build/testapp && ./build/testapp --gpu
```

Run from the application folder: the generated code loads shaders from
`shaders_generated/*.spv` relative to the working directory. Compare `zout_cpu.*` and
`zout_gpu.*`.

## If translation fails

**kslicer exits with code 0 even when clang reports errors while parsing**, and it still writes
(possibly broken) output. Always save the log and check it:

```bash
./cmake-build-release/kslicer ... > kslicer.log 2>&1; grep -n "error:" kslicer.log
```

- Read the kslicer error text: it points to the source line (kernel calling kernel, bad loop
  bound, unsupported vector method, bad reduction form).
- `no member named 'X' in namespace 'std'`: kslicer parses with its own minimal standard
  library `TINYSTL/`, which declares only a subset. Known gap: `std::fabs` (use `std::abs`).
  Present in `TINYSTL/cmath`: `std::min`, `max`, `abs`, `sqrt`, `pow`, `exp`, `log`, `sin`, `cos`,
  `tan`, `asin`, `acos`, `atan`, `atan2`, `floor`, `ceil`, `round` (not `fabs`, `fmin`, `fmax`). Either switch to an
  available function or add the declaration to `TINYSTL/` (an interface is enough).
- A function silently missing from the generated code means it did not match a pattern:
  check the name prefix, the loop shape and that the control function calls a kernel.
- A shader compile error usually disappears after a cosmetic change in the input: split a
  complex expression, add an explicit temporary, avoid a construct from the forbidden list.
- If kslicer can not create the `include` directory in the sample folder, create it manually.
