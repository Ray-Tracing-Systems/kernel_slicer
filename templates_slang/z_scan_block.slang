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

groupshared {{Type}} l_Data[512];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 a_globalTID: SV_DispatchThreadID, uint3 a_localTID: SV_GroupThreadID, uniform KernelArgs kgenArgs)
{
  const uint globalId = a_globalTID[0]; 
  const uint localId  = a_localTID[0];

  {{Type}} idata = {{Type}}(0);
  {{Type}} odata = {{Type}}(0);
  {{Type}} edata = {{Type}}(0);
  
  if(kgenArgs.currMip == 0)
  {
    if (globalId < kgenArgs.iNumElementsX) // don't read in_data[globalId-1] for exclusive scan: in_data may alias out_data (in-place scan), 
      idata = in_data[globalId];          // so neighbour workgroup could have already overwritten it
  }
  else
  {
    if (globalId < kgenArgs.iNumElementsX)
      idata = tmp_data[kgenArgs.currPassOffset + globalId];
  }

  ///////////////////////////////////////////////////// odata = prefix_summ(idata)
  {
    const uint _bsize = 256;
    uint pos = 2 * localId - (localId & (_bsize - 1)); 
    l_Data[pos] = {{Type}}(0);                                      
    pos += _bsize;                                           
    l_Data[pos] = idata;                                     
                                                             
    for (uint offset = 1; offset < _bsize; offset <<= 1)     
    {                                                        
      GroupMemoryBarrierWithGroupSync();
      {{Type}} t = l_Data[pos] + l_Data[pos - offset];          
      GroupMemoryBarrierWithGroupSync();
      l_Data[pos] = t;                                       
    }                                                                                                
    odata = l_Data[pos];
    GroupMemoryBarrierWithGroupSync();
    if(kgenArgs.currMip == 0 && kgenArgs.exclusiveFlag != 0)
      edata = l_Data[pos - 1]; // inclusive summ of previous element; l_Data[_bsize-1] == 0 for localId == 0
  }                        
  /////////////////////////////////////////////////////  

  if(kgenArgs.currMip == 0)
  {
    if (globalId < kgenArgs.iNumElementsX)
      out_data[globalId] = (kgenArgs.exclusiveFlag != 0) ? edata : odata;
  
    if (localId == 255)
      tmp_data[globalId / 256] = odata;
  }
  else
  {
    if (globalId < kgenArgs.iNumElementsX)
      tmp_data[kgenArgs.currPassOffset + globalId] = odata;
  
    if (localId == 255)
      tmp_data[kgenArgs.nextPassOffset + globalId / 256] = odata;
  }
}
