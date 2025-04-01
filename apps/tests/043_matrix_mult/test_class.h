#pragma once
#include "LiteMath.h"
#include <cstdint>

using LiteMath::float4x4;
using LiteMath::float4;

class SimpleTest
{
public:
  SimpleTest();
  
  float4x4 m_mat1;
  float4x4 m_mat2;
  
  virtual void CalcAndAccum(uint32_t a_threadsNum, float* a_out [[size("a_size")]], uint32_t a_size);
  virtual void kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out, uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
