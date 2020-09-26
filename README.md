# kernel_slicer
kernel_slicer

1) sudo apt-get install install cmake

2) sudo apt-get install install ninja-build

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

9) ninja (from build folder of clang)

   the new executable will be build in the "llvm-project/build/bin" folder
   
   you may also used provided VS Code config

 
