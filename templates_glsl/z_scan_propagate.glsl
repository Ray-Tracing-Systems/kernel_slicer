#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, set = 0) buffer data0 { {{Type}} out_data[]; }; //
layout(binding = 1, set = 0) buffer data1 { {{Type}} in_data[]; };

layout( push_constant ) uniform kernelIntArgs
{
  uint iNumElementsX;  
} kgenArgs;

void main()
{
  const uint globalId = gl_GlobalInvocationID[0];
  if(globalId >= kgenArgs.iNumElementsX)
    return;

  const int nextBlockId = int(globalId / 256) - 1;
  if (nextBlockId >= 0)
    out_data[globalId] += in_data[nextBlockId];
}
