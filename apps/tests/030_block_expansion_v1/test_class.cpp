#include "test_class.h"
#include <cstdint>

void TestClass::Test(uint numElements, float* out_buffer)
{
  kernelBE1D_Test<64>(numElements/64, out_buffer);
}
  
template<int bsize>
void TestClass::kernelBE1D_Test(uint blockNum, float* out_buffer)
{
  #pragma omp parallel for
  for(int blockId = 0; blockId < int(blockNum); blockId++) 
  {
    float blockData[bsize];  // will be stored in shared memory

    #pragma omp parallel for
    for(int localId = 0; localId < bsize; localId++) // full parallel
    {
      blockData[localId] = float(localId + blockId);
    }
    
    for(int localId = 0; localId < 4; localId++)                  // single threaded loop
      blockData[localId] = 4.0f;                                        
    
    #pragma omp parallel for 
    for(int localId = 0; localId < bsize; localId++) // full parallel
    {
      out_buffer[blockId*bsize + localId] = blockData[localId];
    }
  }
}