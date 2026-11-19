#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
using LiteMath::uint;

static constexpr uint BLOCK_SIZE = 256;                                   // work group size on GPU
static constexpr uint MAX_SIZE   = BLOCK_SIZE*BLOCK_SIZE*BLOCK_SIZE;      // three levels of scan

// Exclusive prefix sum of an array:
// (1) every level scans blocks of 256 elements and writes the sums of the blocks to the next level;
// (2) offsets of blocks are added back from the top level to the bottom one.
//
class BlockScan
{
public:
  BlockScan()
  {
    m_sums1.resize(MAX_SIZE/BLOCK_SIZE);
    m_offs1.resize(MAX_SIZE/BLOCK_SIZE);
    m_sums2.resize(MAX_SIZE/(BLOCK_SIZE*BLOCK_SIZE));
    m_offs2.resize(MAX_SIZE/(BLOCK_SIZE*BLOCK_SIZE));
    m_sums3.resize(1);
  }

  virtual void ExclusiveScan(const int* a_in [[size("a_size")]], int* a_out [[size("a_size")]], uint a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  template<uint bsize>
  void kernelBE1D_ScanBlocks(uint a_blockNum, const int* a_in, int* a_out, int* a_blockSums, uint a_size);

  template<uint bsize>
  void kernelBE1D_AddOffsets(uint a_blockNum, int* a_data, const int* a_blockOffsets, uint a_size);

  std::vector<int> m_sums1, m_offs1; // sums and offsets of blocks, the first  level
  std::vector<int> m_sums2, m_offs2; // sums and offsets of blocks, the second level
  std::vector<int> m_sums3;          // the total sum
};
