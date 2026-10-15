#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "LiteMath.h"

using LiteMath::float2;

// local variables declared in the single-threaded part of a kernel (before the loop) and used inside the loop:
// they are placed to UBO as '<KernelName>_<varName>'
class KernelLocals
{
public:
  KernelLocals(){}

  void SetWeights(const std::vector<float>& a_weights) { m_weights = a_weights; }

  virtual void Run(const float* a_in [[size("a_size")]], float* a_out [[size("a_size")]], uint32_t a_size, float a_gamma);
  virtual void Run2D(float* a_out [[size("a_width*a_height")]], uint32_t a_width, uint32_t a_height);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Process(const float* a_in, float* a_out, uint32_t a_size, float a_gamma);
  void kernel2D_Gradient(float* a_out, uint32_t a_width, uint32_t a_height);

  float              m_scale  = 2.0f;
  float              m_result = 0.0f;
  std::vector<float> m_weights;
};
