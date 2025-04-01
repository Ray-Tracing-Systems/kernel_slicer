#pragma once

#include <cstdint>

class SimpleTest
{
public:
  SimpleTest() { }

  virtual void CalcAndAccum(uint32_t a_threadsNum, float* a_out [[size("a_size")]], uint32_t a_size);
  virtual void kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out, uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
