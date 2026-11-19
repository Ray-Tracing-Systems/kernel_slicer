#include "test_class.h"

void BlockScan::ExclusiveScan(const int* a_in, int* a_out, uint a_size)
{
  const uint blocks1 = (a_size  + BLOCK_SIZE - 1)/BLOCK_SIZE;
  const uint blocks2 = (blocks1 + BLOCK_SIZE - 1)/BLOCK_SIZE;

  kernelBE1D_ScanBlocks<BLOCK_SIZE>(blocks1, a_in,           a_out,          m_sums1.data(), a_size);
  kernelBE1D_ScanBlocks<BLOCK_SIZE>(blocks2, m_sums1.data(), m_offs1.data(), m_sums2.data(), blocks1);
  kernelBE1D_ScanBlocks<BLOCK_SIZE>(1,       m_sums2.data(), m_offs2.data(), m_sums3.data(), blocks2);

  kernelBE1D_AddOffsets<BLOCK_SIZE>(blocks2, m_offs1.data(), m_offs2.data(), blocks1);
  kernelBE1D_AddOffsets<BLOCK_SIZE>(blocks1, a_out,          m_offs1.data(), a_size);
}

// Exclusive scan inside a block (Hillis-Steele with two halves of shared memory),
// the sum of the block is written to 'a_blockSums[blockId]'.
//
template<uint bsize>
void BlockScan::kernelBE1D_ScanBlocks(uint a_blockNum, const int* a_in, int* a_out, int* a_blockSums, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    int sdata[2*bsize];                                   // shared memory: two halves, read one, write another
    uint src = 0;                                         // local variable of the block: offset of the half to read

    for(uint i = 0; i < bsize; i++) [[parallel]]
    {
      const uint globalId = blockId*bsize + i;
      sdata[i] = (globalId < a_size) ? a_in[globalId] : 0;
    }

    for(uint offset = 1; offset < bsize; offset *= 2)
    {
      for(uint i = 0; i < bsize; i++) [[parallel]]
        sdata[bsize - src + i] = (i >= offset) ? sdata[src + i] + sdata[src + i - offset] : sdata[src + i];
      src = bsize - src;                                  // swap halves
    }

    for(uint i = 0; i < bsize; i++) [[parallel]]          // inclusive ==> exclusive
    {
      const uint globalId = blockId*bsize + i;
      if(globalId < a_size)
        a_out[globalId] = (i == 0) ? 0 : sdata[src + i - 1];
    }

    a_blockSums[blockId] = sdata[src + bsize - 1];         // write to memory: done once per block
  }
}

// Adds the offset of a block to all its elements.
//
template<uint bsize>
void BlockScan::kernelBE1D_AddOffsets(uint a_blockNum, int* a_data, const int* a_blockOffsets, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    const int offset = a_blockOffsets[blockId];           // local variable of the block, read from memory

    for(uint i = 0; i < bsize; i++) [[parallel]]
    {
      const uint globalId = blockId*bsize + i;
      if(globalId < a_size)
        a_data[globalId] += offset;
    }
  }
}
