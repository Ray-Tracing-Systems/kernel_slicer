#version 460
#extension GL_GOOGLE_include_directive : require

#include "include/{{UBOIncl}}"

layout(binding = 0, set = 0) buffer dataUBO { {{MainClassName}}_UBO_Data ubo; };
layout(binding = 1, set = 0) buffer dataInd { uvec4 indirectBuffer[]; };

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
  uvec4 blocksNum = uvec4(1,1,1,0);
  blocksNum.x = ({{Kernel.IndirectSizeX}} + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
  {% if Kernel.threadDim == 2 %}
  blocksNum.y = ({{Kernel.IndirectSizeY}} + {{Kernel.WGSizeY}} - 1)/{{Kernel.WGSizeY}};
  {% endif %}
  {% if Kernel.threadDim == 3 %}
  blocksNum.z = ({{Kernel.IndirectSizeZ}} + {{Kernel.WGSizeZ}} - 1)/{{Kernel.WGSizeZ}};
  {% endif %}
  indirectBuffer[{{Kernel.IndirectOffset}}] = blocksNum;
}