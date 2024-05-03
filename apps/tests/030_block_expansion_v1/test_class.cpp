#include "test_class.h"
#include <cstdint>

void TestClass::Test(uint numElements, float* out_buffer)
{
  kernelBE1D_Test<64>(numElements/64, out_buffer);
}

template<uint bsize>   
void TestClass::kernelBE1D_Test(uint blockNum, float* out_buffer)
{
  //#pragma omp parallel for
  for(uint blockId = 0; blockId < blockNum; blockId++) 
  {
    float blockData[bsize];  // will be stored in shared memory

    for(int localId = 0; localId < bsize; localId++) [[parallel]]  // full parallel
    {
      blockData[localId] = float(localId + blockId);
    }
    
    blockData[5] = 5.0f;
    for(int localId = 0; localId < 4; localId++)                  // single threaded loop
      blockData[localId] = 4.0f;                                        
    
    for(int localId = 0; localId < bsize; localId++) [[parallel]]  // full parallel
    {
      out_buffer[blockId*bsize + localId] = blockData[localId];
    }
  }
}