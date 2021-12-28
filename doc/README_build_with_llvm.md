# Build kernel_slicer as a part of clang/llvm project

1) sudo apt-get install cmake (install cmake)
2) sudo apt-get install ninja-build (install ninja)
3) download build llvm**12** or llvm**14** from source code:
   
   * download source code from https://github.com/llvm/llvm-project/releases/tag/llvmorg-12.0.0
   * cd llvm-project 
   * mkdir build 
   * cd build 
   * cmake -G Ninja -DLLVM_PARALLEL_LINK_JOBS=1 ../llvm -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" 
   * ninja 
   * if it fail, **try to run "ninja" again several times** untill you finally build everithing. 
   * if some parts of llvm are still can not be build, that's not a problem in general
   * for llvm 14 you will need to fix 2 lines in kslicer_main.cpp, around line 500 

4) go to 'llvm-project/clang-tools-extra'

5) git clone --recurse-submodules https://github.com/Ray-Tracing-Systems/kernel_slicer.git
   i.e. put folder on this project project in to "llvm-project/clang-tools-extra" to form "llvm-project/clang-tools-extra/kernel_slicer"
   
6) put "add_subdirectory(kernel_slicer)" to CMakeLists.txt in "clang-tools-extra" folder
7) Rename "CMakeLists.txt" to "CMakeLists2.txt" (please don't commit)
8) Rename "CMakeLists1.txt" to "CMakeLists.txt" (please rename back before commiting!)
9) ninja (**from build folder of clang**)
10) executable will be build in the "llvm-project/build/bin" folder
11) goto "llvm-project/clang-tools-extra/kernel_slicer" directory
12) ../../build/bin/kslicer "apps/05_filter_bloom_good/test_class.cpp" -mainClass ToneMapping -stdlibfolder TINYSTL -pattern ipv -reorderLoops YX -Iapps/LiteMath IncludeToShaders -shaderCC GLSL -DKERNEL_SLICER -v

 
