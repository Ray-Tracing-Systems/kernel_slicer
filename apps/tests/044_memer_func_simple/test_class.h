#pragma once
#include "LiteMath.h"
#include <cstdint>

using LiteMath::float4x4;
using LiteMath::float4;

class SimpleTest
{
public:
  SimpleTest();
  
  float memberFunction(int i);
  
  virtual void CalcAndAccum(uint32_t a_threadsNum, float* a_out [[size("a_threadsNum")]]);
  virtual void kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
