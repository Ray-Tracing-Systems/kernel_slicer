# kernel_slicer: CPU to GPU (Vulkan) translator
<p align = "center"><img src="logo.png" width = "600"></p>
<p align = "center">Fig. 1. Our translator has many heads due to many existing programming patterns.</p><BR>
This project is based on clang (fig. 1.). We sincerely thank the clang front-end developers for the great structure and documentation of their project!

# Project overview

kernel_slicer is auto-programming tool which takes C++ code as input and port this code to GPU by generating Vulkan implemantation automaticly using source-to-source transtation. The current goal of this project is to increase developer productivity when porting CPU code to Vulkan which is time consuming work in general. Please read several short remarks about our project.

* We generate C++ source code in Vulkan. We don't (and don't plan currently) to support any other back-ends (like CUDA or OpenCL) since we see no need for this. Nevertheless, if you see such a need for your project and you like our concept, please contact us;

* Our goal is to generate code as if it were written by hand, but we automate 90% mechanical work which developers have to do when using Vulkan. We don't introduce any ineffitiency in the generated code and generate readable and understandable code. It is assumed that you will use generated code as normal;

* kernel_slicer is NOT a general-purpose programming technology (well, general purpose programming is still possible with it). We use pattern matching to efficiently map certain types of software to GPU. Such types are called patterns;

* Patterns are specific cases of algorithms/software which has known efficient implemantation for GPUs. Because we have additional knowllege about algorithm during translation, we can embede specific optimisation to our translator and leave program logic code clean of these optimisations; 

* Currently we support only one pattern for Ray Tracing. Our next pattern will be for Image Processing; 

* Our tool is not classic compiler. It generate kernels source code for [google clspv](https://github.com/google/clspv "Clspv is a prototype compiler for a subset of OpenCL C to Vulkan compute shaders") and C++ code for Vulkan calls to run kernels correcly. 

* We also keep in mind [Circle shader compiler](https://github.com/seanbaxter/shaders "writing shaders in C++ with Circle compiler") which we are going to use in future as one of our back-ends; 

* Let's summarize again: you have to bind generated code to your program yourself, thus you can't you escape Vulkan experience. This can be done by directly using generated class. You can also override some functions if you want to change behaviour of some generated code parts;

* Our main users are Vulkan developers that has to use Vulkan due to some specific hardware features or performance requirenments. Therefore, we initially pay special attention to interaction between generated and hand written (which can use any desired hardware extensions) code which assumed to be done via inheritance and virtual function overrides;

# Build(1): as stand-alone project

1. sudo apt-get install llvm-10-dev

2. sudo touch /usr/lib/llvm-10/bin/yaml-bench 

3. sudo apt-get install libclang-10-dev 

4. use Cmake and make

5. you may also use provided VS Code config to build and run test cases (tasks.json and launch.json)

6. You will need also to build [google clspv](https://github.com/google/clspv "Clspv is a prototype compiler for a subset of OpenCL C to Vulkan compute shaders").

# Build(2): as a part of llvm project
1. sudo apt-get install cmake

2. sudo apt-get install ninja-build

3. git clone https://github.com/llvm/llvm-project.git 

4. cd llvm-project 
   mkdir build 
   cd build

5. cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra" 

6. ninja

7. go to 'llvm-project/clang-tools-extra'

8. git clone https://github.com/Ray-Tracing-Systems/kernel_slicer.git
   i.e. put folder on this project project in to "llvm-project/clang-tools-extra" to form "llvm-project/clang-tools-extra/kernel_slicer"
   
8. put "add_subdirectory(kernel_slicer)" to CMakeLists.txt in "clang-tools-extra" folder

9. Rename "CMakeLists.txt" to "CMakeLists2.txt"  (please don't commit)

10. Rename "CMakeLists1.txt" to "CMakeLists.txt" (please rename back before commiting!)

11. ninja (from build folder of clang)
     the new executable will be build in the "llvm-project/build/bin" folder
     you may also used provided VS Code config to build and run test cases (tasks1.json and launch1.json)

12. You will need also to build [google clspv](https://github.com/google/clspv "Clspv is a prototype compiler for a subset of OpenCL C to Vulkan compute shaders").

# Concept 

# Patterns

## Ray Tracing Vectorization (RTV) Pattern

In practical applications of Ray Tracing there is a problem with effitiency of complex code: if fact you can't just put all your code for ray/path evaluation to a single kernel if you take care about performance. So, developers usually split their implementation in multiple kernels and pass data between kernels via main memory and may use some optimizations like thread compaction or path regeneration. In the same time, kernels need to access arbitrary data of materials, light and geometry. So, in our opinion RTV template is quite general vectorization template. 

* [Example #1: single kernel ](README_ex00.md)

* [Example #2: passing data between kernels](README_ex01.md)


## Image Processing Vectorization (IPV) Pattern

