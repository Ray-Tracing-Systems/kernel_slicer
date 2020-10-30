# Project overview

![](logo.png =600x450)


# Build(1): as stand-alone project

1) sudo apt-get install llvm-10-dev
2) sudo touch /usr/lib/llvm-10/bin/yaml-bench 
3) sudo apt-get install libclang-10-dev 
4) use Cmake and make
5) you may also use provided VS Code config to build and run test cases (tasks.json and launch.json)

# Build(2): as a part of clang/llvm project

1) sudo apt-get install cmake
2) sudo apt-get install ninja-build
3) git clone https://github.com/llvm/llvm-project.git 
4) cd llvm-project 
   mkdir build 
   cd build

5) cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" 
6) ninja

7) go to 'llvm-project/clang-tools-extra'

8) git clone https://github.com/Ray-Tracing-Systems/kernel_slicer.git
   i.e. put folder on this project project in to "llvm-project/clang-tools-extra" to form "llvm-project/clang-tools-extra/kernel_slicer"
   
8) put "add_subdirectory(kernel_slicer)" to CMakeLists.txt in "clang-tools-extra" folder
9) Rename "CMakeLists.txt" to "CMakeLists2.txt"  (please don't commit)
10) Rename "CMakeLists1.txt" to "CMakeLists.txt" (please rename back before commiting!)
11) ninja (from build folder of clang)
     the new executable will be build in the "llvm-project/build/bin" folder
     you may also used provided VS Code config to build and run test cases (tasks1.json and launch1.json)

 
