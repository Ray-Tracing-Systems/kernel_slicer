#include "test_class.h"
#include <cstdint>

void TestClass::Test(uint numElements, float* out_buffer)
{
  kernelBE1D_Test<64>(numElements/64, out_buffer);
}
  
template<int bsize>
void TestClass::kernelBE1D_Test(uint blockNum, float* out_buffer)
{
  for(int blockId = 0; blockId < int(blockNum); blockId++) 
  {
    float blockData[bsize];  // will be stored in shared memory

    [[parallel]] for(int localId = 0; localId < bsize; localId++) // full parallel
    {
      blockData[localId] = float(localId + blockId);
    }
    
    [[parallel]] for(int localId = 0; localId < bsize; localId++) // single thread per block, simple loop (!!!)
    {
      out_buffer[blockId*bsize + localId] = blockData[localId];
    }
  }
}