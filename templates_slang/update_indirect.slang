{% include "inc_ubo.slang" %}
RWStructuredBuffer<{{MainClassName}}{{MainClassSuffix}}_UBO_Data> ubo;
RWStructuredBuffer<uint4> indirectBuffer;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
  uint4 blocksNum = uint4(1,1,1,0);
  blocksNum.x = ({{Kernel.IndirectSizeX}} - {{Kernel.IndirectStartX}} + {{Kernel.WGSizeX}} - 1)/{{Kernel.WGSizeX}};
  {% if Kernel.threadDim == 2 %}
  blocksNum.y = ({{Kernel.IndirectSizeY}} - {{Kernel.IndirectStartY}} + {{Kernel.WGSizeY}} - 1)/{{Kernel.WGSizeY}};
  {% endif %}
  {% if Kernel.threadDim == 3 %}
  blocksNum.z = ({{Kernel.IndirectSizeZ}} - {{Kernel.IndirectStartZ}} + {{Kernel.WGSizeZ}} - 1)/{{Kernel.WGSizeZ}};
  {% endif %}
  indirectBuffer[{{Kernel.IndirectOffset}}] = blocksNum;
}