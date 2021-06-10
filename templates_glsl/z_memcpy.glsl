#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, set = 0) buffer data0 { float out_data[]; }; //
layout(binding = 1, set = 0) buffer data1 { float in_data[]; };

layout( push_constant ) uniform kernelIntArgs
{
  uint iNumElementsX;  
} kgenArgs;

void main()
{
  const uint i = gl_GlobalInvocationID[0]; 
  if(i >= kgenArgs.iNumElementsX)
    return;
  out_data[i] = in_data[i];
}
