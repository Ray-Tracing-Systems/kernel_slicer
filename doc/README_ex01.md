
# Example #2: passing data between kernels

The kernel_slicer can be thought of a sort of high level vectorization technology (fig. 2) which uses Vulkan as a core back-end for actual code vectorization. In fact we have many existing vectorization technologies like CUDA, OpenCL, ISPC, Vulkan and some other like [enoki](https://github.com/mitsuba-renderer/enoki "structured vectorization and differentiation on modern processor architectures"), TBD add more ...

<p align = "center"><img src="images/vector_inst.png" width = "500" align = "center"></p><p align = "center">Fig. 2. Code vectorization technology idea</p><BR>

When you are exploring different ways of splitting your code into pieces to better optimize those pieces with a vectorizing compiler, you have to try many different variants which is time consuming with Vulkan and make optimization process hard.

To adress this issue within RTV pattern you may just write scalar code inside control functions (and inside kernels also!) and declare variables which will be vectorized automaticly in generated code. When you declare scalar variable inside control function the translator will generate separate buffer for store intermediate data which you use to pass data between kernels.

```cpp
float  data1; ------> float data1_V[N];
float4 data4; ------> float data4_V[N];
```

Or in Vulkan:
```cpp
float  data1; ------> VkBuffer data1Buffer;
float4 data4; ------> VkBuffer data4Buffer;
```

So, consider further control function:

```cpp
void TestClass::MainFunc(uint tidX, uint tidY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tidX, tidY, 
                    &rayPosAndNear, &rayDirAndFar); // ==> (rayPosAndNear, rayDirAndFar)

  Lite_Hit hit;
  kernel_RayTrace(tidX, tidY, &rayPosAndNear, &rayDirAndFar, 
                  &hit); // (rayPosAndNear, rayDirAndFar) ==> hit
  
  kernel_TestColor(tidX, tidY, &hit, 
                   out_color);  // hit ==> out_color
}
```

Here we have three local variables (*rayPosAndNear, rayDirAndFar, hit*) which will be translated to buffers and three kernel functions which will be translated to kernels. It is **developer responsibility** to define how to split code to kernels, but in general this is should be consistent with splitting your program logic to functions in normal CPU code.

You can imagine local variables of the control function as being placed on the stack (which in general is happened for CPU code). And think about the code that is implemented inside the kernels --- as about highly optimized code that mostly uses registers and stores the result on the stack in these variables.

Now let us see input and generated source code for kernel_InitEyeRay:
```cpp
void TestClass::kernel_InitEyeRay(uint tid, float4* rayPosAndNear, float4* rayDirAndFar)
{
  const float3 rayDir = EyeRayDir(tid, /* ... */, m_worldViewProjInv); 
  const float3 rayPos = make_float3(0.0f, 0.0f, 0.0f);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}
```
And the generated kernel:

```cpp
__kernel void kernel_InitEyeRay(
  __global float4 * restrict rayPosAndNear,
  __global float4 * restrict rayDirAndFar,
  __global const uint* restrict kgen_data,
  const uint kgen_iNumElementsX,
  const uint kgen_iNumElementsY)
{
  /////////////////////////////////////////////////
  const uint tid = get_global_id(0);
  if (tid >= kgen_iNumElementsX)
    return;
  /////////////////////////////////////////////////

  const float3 rayDir = EyeRayDir(tid, /* ... */, *(  (__global const float4x4*)(kgen_data+MATRIX_OFFSET)  )); 
  const float3 rayPos = make_float3(0.0f, 0.0f, 0.0f);
  
  rayPosAndNear[tid] = to_float4(rayPos, 0.0f);
  rayDirAndFar [tid] = to_float4(rayDir, FLT_MAX);
}
```

Here you can see several things happened:

1. For all local variables of control functions that were **passed to kernels by address** , thread offsets were added;

2. For class member m_worldViewProjInv which is accesed inside "TestClass::kernel_InitEyeRay" the code was changed to access this member via data buffer 'kgen_data' at particular offset. In the generated Vulkan code we will also have method to update m_worldViewProjInv at MATRIX_OFFSET.

Here you saw that inside kernel functions you can access data class member *m_worldViewProjInv* which is understanded by the translator and transformed to appropriate GPU code. In the following examples, you will see that you can directly access std::vector<...> data members of main class inside kernels. Also you can call member functions from kernels (**WORK IN PROGRESS!**) which turns GPU programming to just writing common object oriented code in C++ for some cases.

Now let us see what happends inside generated control function:


```cpp
void TestClass_Generated::MainFuncCmd(VkCommandBuffer a_commandBuffer, int tidX, uint tidY, uint* out_color)
{
  m_currCmdBuffer = a_commandBuffer;

  float4 rayPosAndNear, rayDirAndFar;
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, InitEyeRayLayout, 0, 1, &m_allGeneratedDS[0], 0, nullptr);
  InitEyeRayCmd(tidX, tidY, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit;
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RayTraceLayout, 0, 1, &m_allGeneratedDS[1], 0, nullptr);
  RayTraceCmd(tidX, tidY, &rayPosAndNear, &rayDirAndFar, 
              &hit);
  
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, TestColorLayout, 0, 1, &m_allGeneratedDS[2], 0, nullptr);
  TestColorCmd(tidX, tidY, &hit, 
               out_color);
}
```

Since we use automatic source-to-source translation, you shouldn't be confused with old variables (*rayPosAndNear, rayDirAndFar, hit*) which are still presented but not used this time. Here we have to pay attention to 2 things. First, the control flow has been preserved. Second, for each kernel call specific descriptor set was initialized and *vkCmdBindDescriptorSets*  is inserted before kernel calls to bind corrent input and ouptut buffers to each kernel.

The full source code of this example is located at [apps/01_intersectSphere/](apps/01_intersectSphere/). The example is relatively simple, it generate ray in first kernel, intersect sphere in the second one and then finally put color to resulting *out_color* in the third one. 

