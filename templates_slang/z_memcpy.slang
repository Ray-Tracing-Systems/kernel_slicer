//

RWStructuredBuffer<float> out_data;  
RWStructuredBuffer<float> in_data; 

struct KernelArgs
{
  uint iNumElementsX;  
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 a_globalTID: SV_DispatchThreadID, uniform KernelArgs kgenArgs)
{
  const uint i = a_globalTID[0]; 
  if(i >= kgenArgs.iNumElementsX)
    return;
  out_data[i] = in_data[i];
}
