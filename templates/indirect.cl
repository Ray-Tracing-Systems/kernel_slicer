#include "include/{{UBOIncl}}"

__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void DummyKernel(__global uint4* indirectBuffer)
{
  uint4 blocksNum = {0,0,0,0};
  indirectBuffer[0] = blocksNum;
}

## for Kernel in Kernels
__attribute__((reqd_work_group_size(1, 1, 1)))
__kernel void {{Kernel.Name}}_UpdateIndirect(__global struct {{MainClassName}}_UBO_Data* restrict ubo, __global uint4* indirectBuffer)
{
  uint4 blocksNum = {0,0,0,0};
  blocksNum.x = ({{Kernel.IndirectSizeX}} + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
  {% if Kernel.threadDim == 2 %}
  blocksNum.y = ({{Kernel.IndirectSizeY}} + {{Kernel.WGSizeY}} - 1)/{{Kernel.WGSizeY}};
  {% endif %}
  {% if Kernel.threadDim == 3 %}
  blocksNum.z = ({{Kernel.IndirectSizeZ}} + {{Kernel.WGSizeZ}} - 1)/{{Kernel.WGSizeZ}};
  {% endif %}
  indirectBuffer[{{Kernel.IndirectOffset}}] = blocksNum;
} 
## endfor