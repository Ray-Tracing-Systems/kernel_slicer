# kernel_slicer: CPU to GPU (Vulkan) translator
<img src="logo.png" width = "600" align = "center">
<BR>This project is based on clang. We sincerely thank the clang front-end developers for the great structure and documentation of their project!

# Project overview

kernel_slicer is auto-programming tool which takes C++ code as input and port this code to GPU by generating Vulkan implemantation automaticly using source-to-source transtation. The current goal of this project is to increase developer productivity when porting CPU code to Vulkan which is time consuming work in general. Please read several short remarks about our project.

* We generate C++ source code in Vulkan. We don't (and don't plan currently) to support any other back-ends (like CUDA or OpenCL) since we see no need for this. Nevertheless, if you see such a need for your project and you like our concept, please contact us;

* Our goal is to generate code as if it were written by hand, but we automate 90% mechanical work which developers have to do when using Vulkan. We don't introduce any ineffitiency in the generated code and generate readable and understandable code. It is assumed that you will use generated code as normal;

* kernel_slicer is NOT a general-purpose programming technology (well, general purpose programming is still possible with it). We use pattern matching to efficiently map certain types of software to GPU. Such types are called patterns;

* Patterns are specific cases of algorithms/software which has known efficient implemantation for GPUs. Because we have additional knowllege about algorithm during translation, we can embede specific optimisation to our translator and leave program logic code clean of these optimisations; 

* Currently we support only one pattern for Ray Tracing. Our next pattern will be for Image Processing; 

* Our tool is not classic compiler. It generate kernels source code for [google clspv](https://github.com/google/clspv "Clspv is a prototype compiler for a subset of OpenCL C to Vulkan compute shaders") and C++ code for Vulkan calls to run kernels correcly. 

* We also keep in mind [Circle shader compiler](https://github.com/seanbaxter/shaders "writing shaders in C++ with Circle compiler") which we are going to use in future as one of our back-ends; 

* Let's summarize again: you have to bind generated code to your program yourself, thus you can't you can't escape Vulkan experience. This can be done by directly using generated class. You can also override some functions if you want to change behaviour of some generated code parts;

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

### Example #1: single kernel 

Let us first consider how to generate single kernel launch with our technology.

```cpp
class TestClass 
{
public:
  void PackXY(uint tidX, uint tidY, uint* out_pakedXY);
  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);
};

void TestClass::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel_PackXY(tidX, tidY, out_pakedXY);
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[tidY*WIN_WIDTH+tidX] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

```

Which is straitforward C++ code and can be normally used in C++:
```cpp
std::vector<uint32_t> packedXY(WIN_WIDTH*WIN_HEIGHT);
for(int y=0;y<WIN_HEIGHT;y++)
  for(int x=0;x<WIN_WIDTH;x++)
    test.PackXY(x, y, packedXY.data()); // remember pitch-linear (x,y) 
```

Ok, let us just immediately show the generated code (well, most interesting part of it) for such a simple case. For kernel:

```cpp
__kernel void kernel_PackXY(
  __global uint* out_pakedXY,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY)
{
  /////////////////////////////////////////////////////////////////
  const uint tidX = get_global_id(0); 
  const uint tidY = get_global_id(1); 
  if(tidX >= kgen_iNumElementsX || tidY >= kgen_iNumElementsY)
    return;
  /////////////////////////////////////////////////////////////////
  out_pakedXY[tidY*WIN_WIDTH+tidX] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}
```
And the generated Vulkan code:
```cpp
void TestClass_Generated::PackXYCmd(uint tidX, uint tidY, uint* out_pakedXY)
{
  vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, PackXYPipeline);
  uint32_t pcData[3] = { tidX, tidY, 1 };
  vkCmdPushConstants(m_currCmdBuffer, PackXYLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)*3, pcData);
  vkCmdDispatch(m_currCmdBuffer, tidX/m_blockSize[0], tidY/m_blockSize[1], 1/m_blockSize[2]);
  //and the barrier, not shown here 
}

void TestClass_Generated::PackXYCmd(VkCommandBuffer a_commandBuffer, int tidX, uint tidY, uint* out_pakedXY)
{
  m_currCmdBuffer = a_commandBuffer;
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, PackXYLayout, 0, 1, &m_allGeneratedDS[0], 0, nullptr);
  PackXYCmd(tidX, tidY, out_pakedXY);
}
```

Now let's analyze this example. In this example we used three **special names**:

1. "tidX";

2. "tidY";

3. Prefix "kernel_".

First two indicates that arguments of both "kernel_PackXY" and "PackXY" are considered by kernel_slicer in a special way: as thread id. The last one prompt our translator to threat "kernel_PackXY" function in a special way: as seperate a kernel. 

Fine, so, **why do we need this second PackXY class member?**

The answer is that PackXY tells to generator what kernels you actually want to run and in which order. We call such functions "Main" or "Control" function because they control kernel execution order. So, basically RTV template consists of 2 types of functions:

1. Kernels which always should have prefix "kernel_" in their name;

2. Control functions, which define sequence kernels that will be placed in command buffer. It should be note immediately that **control functions are restricted in several ways**. These restrictions are actually define what you can do within the RTV pattern and what you can't. We will consider them later during more complex examples.

Now it is clear why 2 functions were generated. The first one reflects single kernel run for "kernel_PackXY":
```cpp
void TestClass_Generated::PackXYCmd(uint tidX, uint tidY, uint* out_pakedXY);
```
In general, you should not call this function in you code.

The second one reflects contrtol function and it assumed you will use it:
```cpp
void TestClass_Generated::PackXYCmd(VkCommandBuffer a_commandBuffer, int tidX, uint tidY, uint* out_pakedXY)
```
In fact this is not the end because there is a question of where in GPU memory the result of this last PackXYCmd will be stored. Yes, we have missed this thing previously. So, one more function will be generated actually:

```cpp
void TestClass_Generated::SetVulkanInOutFor_PackXY(VkBuffer a_out_pakedXYBuffer,   
                                                   size_t   a_out_pakedXYOffset);
```

Well, Vulkan separate input/output definition for you kernels and kernel lauch logic which is defined during command buffer write. During writing command buffer you can't actually create new binding for you kernel. From the Vulkan specification: The descriptor set contents bound by a call to vkCmdBindDescriptorSets may be consumed during host execution of the command, or during shader execution of the resulting draws, or any time in between. Thus, the contents must not be altered (overwritten by an update command, or freed) between when the command is recorded and when the command completes executing on the queue. 


Thus, you must create all bindings in advance for which we generate these "SetVulkanInOutFor_XXX" functions. 

### Example #2: passing data between kernels

## Image Processing Vectorization (IPV) Pattern

