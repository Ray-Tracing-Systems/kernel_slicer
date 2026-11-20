#include "test_class.h"

using LiteMath::InterlockedAdd;

void Histogram::Compute(const uint* a_in, uint* a_hist, uint a_size)
{
  kernel1D_Clear(NUM_BINS, a_hist);
  kernelBE1D_Histogram<BLOCK_SIZE>(DivUp(a_size, ELEMS_PER_BLOCK), a_in, a_hist, a_size);
}

void Histogram::kernel1D_Clear(uint a_size, uint* a_data)
{
  for(uint i = 0; i < a_size; i++)
    a_data[i] = 0;
}

template<uint bsize>
void Histogram::kernelBE1D_Histogram(uint a_blockNum, const uint* a_in, uint* a_hist, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    uint hist[NUM_BINS];                                  // histogram of the block, shared memory

    for(uint i = 0; i < NUM_BINS; i++) [[parallel]]
      hist[i] = 0;

    for(uint i = 0; i < ELEMS_PER_BLOCK; i++) [[parallel]]
    {
      const uint globalId = blockId*ELEMS_PER_BLOCK + i;
      if(globalId < a_size)
        InterlockedAdd(hist[a_in[globalId] % NUM_BINS], 1u); // atomic in shared memory
    }

    for(uint i = 0; i < NUM_BINS; i++) [[parallel]]
    {
      if(hist[i] != 0)
        InterlockedAdd(a_hist[i], hist[i]);               // atomic in global memory: histograms of blocks are added
    }
  }
}
