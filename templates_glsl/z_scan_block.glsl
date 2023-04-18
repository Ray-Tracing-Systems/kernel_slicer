#version 460
layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, set = 0) buffer data0 { {{Type}} out_data[]; }; //
layout(binding = 1, set = 0) buffer data1 { {{Type}} in_data[]; };

layout( push_constant ) uniform kernelIntArgs
{
  uint iNumElementsX;  
} kgenArgs;

shared {{Type}} l_Data[512];

void main()
{
  const uint globalId = gl_GlobalInvocationID[0]; 
  const uint localId  = gl_LocalInvocationID[0];

  {{Type}} idata = {{Type}}(0);
  {{Type}} odata = {{Type}}(0);

  if (globalId < kgenArgs.iNumElementsX)
    idata = in_data[globalId];
  
  ///////////////////////////////////////////////////// odata = prefix_summ(idata)
  {
    const uint _bsize = 256;
    uint pos = 2 * localId - (localId & (_bsize - 1)); 
    l_Data[pos] = {{Type}}(0);                                      
    pos += _bsize;                                           
    l_Data[pos] = idata;                                     
                                                             
    for (uint offset = 1; offset < _bsize; offset <<= 1)     
    {                                                        
      barrier();                                    
      {{Type}} t = l_Data[pos] + l_Data[pos - offset];          
      barrier();                                     
      l_Data[pos] = t;                                       
    }                                                                                                
    odata = l_Data[pos];         
  }                        
  /////////////////////////////////////////////////////  

  if (globalId < kgenArgs.iNumElementsX)
    in_data[globalId] = odata;

  if (localId == 255)
    out_data[globalId / 256] = odata;
}
