[[vk::binding(0, 0)]] RWStructuredBuffer<{{Type}}> in_data;
[[vk::binding(1, 0)]] RWStructuredBuffer<{{Type}}> out_data;
[[vk::binding(2, 0)]] RWStructuredBuffer<{{Type}}> tmp_data;

struct KernelArgs
{
  uint iNumElementsX;  
  uint currMip;
  uint currPassOffset;
  uint nextPassOffset;
  uint exclusiveFlag;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 a_globalTID: SV_DispatchThreadID, uniform KernelArgs kgenArgs)
{
  const uint globalId = a_globalTID[0];
  if(globalId >= kgenArgs.iNumElementsX)
    return;
  
  const int nextBlockId = int(globalId / 256) - 1;

  if(kgenArgs.currMip == 0)
  {
    if (nextBlockId >= 0)
      out_data[globalId] += tmp_data[nextBlockId];
  }
  else
  {
    if (nextBlockId >= 0)
      tmp_data[kgenArgs.nextPassOffset + globalId] += tmp_data[kgenArgs.currPassOffset + nextBlockId];
  }
}
