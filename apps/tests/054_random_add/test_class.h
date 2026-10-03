#pragma once

#include <cstdint>
#include "LiteMath.h"
using LiteMath::float3;

class RandomAdd
{
public:
  RandomAdd() { }

  // a_out: a_size pixels with 4 floats per pixel, only first 3 channels are accumulated;
  // a_sameNum consecutive iterations add to the same pixel (1 -- every iteration goes to random pixel)
  virtual void AddOnes(uint32_t a_iterNum, float* a_out [[size("a_size*4")]], uint32_t a_size, uint32_t a_sameNum);
  virtual void kernel1D_AddOnes(uint32_t a_iterNum, float* a_out, uint32_t a_size, uint32_t a_sameNum);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};

static inline uint32_t PseudoRandomIndex(uint32_t i, uint32_t a_size) // LCG step from iteration number with high bits mixed into low ones
{
  const uint32_t x = i * 1103515245u + 12345u;
  return (x ^ (x >> 16)) % a_size;
}
