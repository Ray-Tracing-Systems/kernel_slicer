# Building kernel_slicer

Requirements: CMake ≥ 3.8, a C++17 compiler, **LLVM/clang 17** (default) or **22.1**
(optional, development libraries), and the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home), which provides `slangc` and `glslangValidator`.

```bash
git clone --recurse-submodules https://github.com/Ray-Tracing-Systems/kernel_slicer
cd kernel_slicer
# if cloned without submodules: git submodule update --init
```

## Linux with apt (Ubuntu, Debian)

```bash
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 17
sudo apt install llvm-17-dev libclang-17-dev clang-17
sudo touch /usr/lib/llvm-17/bin/yaml-bench

mkdir cmake-build-release && cd cmake-build-release
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 10
```

### Optional LLVM/Clang 22.1 configuration

Install the development packages alongside LLVM 17:

```bash
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && sudo ./llvm.sh 22
sudo apt install llvm-22-dev libclang-22-dev libclang-cpp22-dev clang-22
```

Configure a separate build directory:

```bash
cmake -S . -B cmake-build-llvm22 -DCMAKE_BUILD_TYPE=Release \
      -DKSLICER_USE_LLVM22=ON \
      -DLLVM_DIR=/usr/lib/llvm-22/lib/cmake/llvm \
      -DClang_DIR=/usr/lib/llvm-22/lib/cmake/clang
cmake --build cmake-build-llvm22 -j 10
```

`KSLICER_USE_LLVM22` defaults to `OFF`, selecting LLVM/Clang 17. When enabled, it
selects LLVM/Clang 22.1 and defines `KSLICER_USE_LLVM22=1` for the `kslicer` target;
C++ code can use `#ifdef KSLICER_USE_LLVM22` for version-specific API changes.
The define is absent in the default LLVM 17 configuration. Use separate build
directories for the two versions to keep their CMake package paths and object files separate.

## Other Linux distributions (Manjaro, Arch, ...)

Use the prebuilt LLVM 17 from the LLVM releases and build kslicer with its clang:

```bash
curl -LO https://github.com/llvm/llvm-project/releases/download/llvmorg-17.0.6/clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz
sudo mkdir -p /opt/llvm17
sudo tar -xf clang+llvm-17.0.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz -C /opt/llvm17 --strip-components=1
sudo touch /opt/llvm17/bin/yaml-bench

cd kernel_slicer
mkdir cmake-build-release && cd cmake-build-release
cmake -S .. -B . \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_CXX_FLAGS="-include cstdint -fno-rtti" \
      -DLLVM_DIR=/opt/llvm17/lib/cmake/llvm \
      -DClang_DIR=/opt/llvm17/lib/cmake/clang
make -j 10
```

## Windows

1. Build LLVM 17 with clang ([instructions](https://llvm.org/docs/GettingStarted.html#getting-the-source-code-and-building-llvm)):
   ```
   git clone --branch release/17.x --depth=1 https://github.com/llvm/llvm-project.git
   cd llvm-project
   cmake -S llvm -B build -G "Visual Studio 17 2022" -DLLVM_ENABLE_PROJECTS="clang" -DCMAKE_BUILD_TYPE=Release
   ```
   Open `build/LLVM.sln` and build the Release configuration. If the build stops because the
   compiler runs out of heap memory, start it again: it continues from where it stopped.
2. Build kernel_slicer:
   ```
   mkdir cmake-build-release && cd cmake-build-release
   cmake -DLLVM_DIR=..\llvm-project\build\lib\cmake\llvm -DClang_DIR=..\llvm-project\build\lib\cmake\clang -DCMAKE_BUILD_TYPE=Release ..
   ```
   Open `kernel_slicer.sln` and build the Release configuration.

You can also [build kslicer as part of LLVM](build_with_llvm.md).

## Libraries for the samples

```bash
cd apps/volk && cmake . && make && cd ../..   # Vulkan loader used by all samples
sudo apt install libfreeimage-dev libglfw3-dev  # needed by some samples only
```

## Optional tools

- [Intel ISPC](https://ispc.github.io/), for the `ispc` back end (CPU vectorization);
- the CUDA or HIP SDK, for the `cuda` / `hip` back ends.

The repository has VS Code configurations (`.vscode/tasks.json`, `.vscode/launch*.json`) to build
kslicer and translate every sample and test.
