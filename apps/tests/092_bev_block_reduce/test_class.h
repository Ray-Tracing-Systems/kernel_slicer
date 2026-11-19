#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
using LiteMath::uint;

static constexpr uint BLOCK_SIZE = 256;                                   // work group size on GPU
static constexpr uint MAX_SIZE   = BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;      // three levels of reduction

// Sum of an array with three passes of block reduction: a block of 256 elements gives one partial sum.
//
class BlockReduce
{
public:
  BlockReduce()
  {
    m_sums1.resize(MAX_SIZE/BLOCK_SIZE);
    m_sums2.resize(MAX_SIZE/(BLOCK_SIZE*BLOCK_SIZE));
  }

  virtual void Reduce(const int* a_in [[size("a_size")]], int* a_out [[size("1")]], uint a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  template<uint bsize>
  void kernelBE1D_ReduceBlocks(uint a_blockNum, const int* a_in, int* a_out, uint a_size);

  std::vector<int> m_sums1; // partial sums of the first  pass
  std::vector<int> m_sums2; // partial sums of the second pass
};
