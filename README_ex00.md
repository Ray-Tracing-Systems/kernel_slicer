# Example #1: single kernel 

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

