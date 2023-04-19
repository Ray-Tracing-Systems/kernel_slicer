#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, set = 0) buffer data0 { {{Type}} in_data []; }; 
layout(binding = 1, set = 0) buffer data1 { {{Type}} out_data[]; };
layout(binding = 2, set = 0) buffer data2 { {{Type}} tmp_data[]; };

layout( push_constant ) uniform kernelIntArgs
{
  uint iNumElementsX;  
  uint currMip;
  uint currPassOffset;
  uint nextPassOffset;
} kgenArgs;

void main()
{
  const uint globalId = gl_GlobalInvocationID[0];
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
