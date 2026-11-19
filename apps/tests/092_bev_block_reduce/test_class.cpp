#include "test_class.h"

void BlockReduce::Reduce(const int* a_in, int* a_out, uint a_size)
{
  const uint blocks1 = (a_size  + BLOCK_SIZE - 1)/BLOCK_SIZE;
  const uint blocks2 = (blocks1 + BLOCK_SIZE - 1)/BLOCK_SIZE;
  kernelBE1D_ReduceBlocks<BLOCK_SIZE>(blocks1, a_in,           m_sums1.data(), a_size);
  kernelBE1D_ReduceBlocks<BLOCK_SIZE>(blocks2, m_sums1.data(), m_sums2.data(), blocks1);
  kernelBE1D_ReduceBlocks<BLOCK_SIZE>(1,       m_sums2.data(), a_out,          blocks2);
}

// One block sums 'bsize' elements of 'a_in' and writes the sum to 'a_out[blockId]'.
//
template<uint bsize>
void BlockReduce::kernelBE1D_ReduceBlocks(uint a_blockNum, const int* a_in, int* a_out, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    int sdata[bsize];                                     // shared memory of the block

    for(uint i = 0; i < bsize; i++) [[parallel]]          // load, zero outside of the array
    {
      const uint globalId = blockId*bsize + i;
      sdata[i] = (globalId < a_size) ? a_in[globalId] : 0;
    }

    for(uint stride = bsize/2; stride > 0; stride /= 2)   // ordinary loop with a parallel loop inside
    {
      for(uint i = 0; i < stride; i++) [[parallel]]
        sdata[i] += sdata[i + stride];
    }

    a_out[blockId] = sdata[0];                            // write to memory: done once per block
  }
}
