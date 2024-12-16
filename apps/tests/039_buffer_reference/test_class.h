#pragma once
#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"

using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;

class VectorTest
{
public:
  VectorTest(size_t a_size);

  virtual void Test(float* a_data [[size("a_size")]], unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  std::vector<float> m_data1;
  std::vector<float> m_data2;
};
