#include "test_class.h"
#include <cstdint>

void TestClass::Test(uint numElements, float* out_buffer)
{
  kernelBE1D_Test(numElements/MY_BLOCK_SIZE, out_buffer);
}
  
void TestClass::kernelBE1D_Test(uint blockNum, float* out_buffer)
{
  //#pragma omp parallel for
  for(uint blockId = 0; blockId < blockNum; blockId++) 
  {
    float blockData[MY_BLOCK_SIZE];  // will be stored in shared memory

    for(int localId = 0; localId < MY_BLOCK_SIZE; localId++) [[parallel]]  // full parallel
    {
      blockData[localId] = float(localId + blockId);
    }
    
    blockData[5] = 5.0f;
    for(int localId = 0; localId < 4; localId++)                  // single threaded loop
      blockData[localId] = 4.0f;                                        
    
    for(int localId = 0; localId < MY_BLOCK_SIZE; localId++) [[parallel]]  // full parallel
    {
      out_buffer[blockId*MY_BLOCK_SIZE + localId] = blockData[localId];
    }
  }
}