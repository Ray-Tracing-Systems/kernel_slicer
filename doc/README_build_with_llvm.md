# Build `kernel_slicer` as a part of Clang/LLVM project

## Brief

`kernel_slicer` can be built within LLVM project with [build script](/other/build-with-llvm.sh).

## Installation process

### Description

This script will download the source code of LLVM project and integrate `kernel_slicer` into it.

### Installation steps

Download LLVM project and integrate `kernel_slicer` in it with prepared bash script:

```shell
# FIXME(hack3rmann): provide link from master branch
wget https://raw.githubusercontent.com/Ray-Tracing-Systems/kernel_slicer/refs/heads/fix/build-with-llvm-update/doc/other/build-with-llvm.sh
chmod +x build-with-llvm.sh
./build-with-llvm.sh
```

Build LLVM project with `kernel_slicer`:

```shell
cd llvm-project
cmake -G Ninja -B build -S llvm -DLLVM_PARALLEL_LINK_JOBS=1 \
                                -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" \
                                -DCMAKE_BUILD_TYPE=Release \
                                -DKSLICER_LLVM_BUILD=1
ninja -C build
```

Note that `KSLICER_LLVM_BUILD` should be defined to make `kslicer` a part of LLVM.

## Executable

After build you can find executable `kslicer` in `build/bin/` directory of `llvm-project`.

## Sanity check

You can run a test to check that `kslicer` works properly:

```shell
cd clang-tools-extra/kernel_slicer
../../build/bin/kslicer "apps/05_filter_bloom_good/test_class.cpp" \
    -mainClass ToneMapping -stdlibfolder TINYSTL -pattern ipv -reorderLoops YX -Iapps/LiteMath IncludeToShaders -shaderCC GLSL -DKERNEL_SLICER -v
```
