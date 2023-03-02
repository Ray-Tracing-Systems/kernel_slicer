#pragma once
#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"

using LiteMath::float4x4;
using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;
using LiteMath::as_uint;
using LiteMath::as_float;
using LiteMath::half4;

struct TestBox
{
  half4 b1;
  half4 b2;
  half4 b3;
  uint32_t offs1;
  uint32_t offs2;
};

class Padding
{
public:
  Padding()
  { 
    m_data.resize(4);
    for(size_t i=0;i < m_data.size();i++) {
      float4 a = float4(i*8 + 0, i*8 + 1, i*8 + 2, i*8 + 3);
      float4 b = float4(i*8 + 4, i*8 + 5, i*8 + 6, i*8 + 7);
      m_data[i].b1 = half4(a);
      m_data[i].b2 = half4(b);
      m_data[i].b3 = half4((a-b)/(a+b));
      m_data[i].offs1 = i*2+1;
      m_data[i].offs2 = i*3+7;
    }

    std::cout << "sizeof(half4)   = " << sizeof(half4)   << std::endl;
    std::cout << "sizeof(TestBox) = " << sizeof(TestBox) << std::endl;
  }

  virtual void Test(float* a_data __attribute__((size("a_size*14"))), unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);

  std::vector<TestBox> m_data;

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
