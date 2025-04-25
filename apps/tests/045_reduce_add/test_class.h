#pragma once
#include "LiteMath.h"
#include <cstdint>
#include <vector>

class SimpleTest
{
public:
  SimpleTest();
  
  virtual void CalcAndAccum(const float* in_data [[size("a_threadsNum")]], uint32_t a_threadsNum, 
                            float* a_outAccum    [[size("5")]]);
  virtual void kernel1D_CalcAndAccum(const float* in_data , uint32_t a_threadsNum, float* a_out);
  virtual void kernel1D_CopyData(float* a_out, const float* a_in, uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
protected:
  std::vector<float> m_accum;
};
