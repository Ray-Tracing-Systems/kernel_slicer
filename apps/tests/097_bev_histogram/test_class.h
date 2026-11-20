#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
using LiteMath::uint;

static constexpr uint NUM_BINS        = 256;
static constexpr uint BLOCK_SIZE      = 256;            // work group size on GPU
static constexpr uint ELEMS_PER_BLOCK = BLOCK_SIZE*16;  // elements processed by a block

static inline uint DivUp(uint a, uint b) { return (a + b - 1)/b; }

// Histogram of values: every block counts its elements in shared memory with atomics,
// then adds its histogram to the result with atomics in global memory.
//
class Histogram
{
public:
  Histogram(){}

  virtual void Compute(const uint* a_in [[size("a_size")]], uint* a_hist [[size("NUM_BINS")]], uint a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  virtual void kernel1D_Clear(uint a_size, uint* a_data);

  template<uint bsize>
  void kernelBE1D_Histogram(uint a_blockNum, const uint* a_in, uint* a_hist, uint a_size);
};
