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

struct BVHNode 
{
  float3   boxMin;
  uint32_t leftOffset;
  float3   boxMax;
  uint32_t escapeIndex;
  //float4 boxMin;
  //float4 boxMax;
};

class Padding
{
public:
  Padding()
  { 
    m_data.resize(4);
    for(size_t i=0;i < m_data.size();i++) {
      m_data[i].boxMin      = float3(i*8 + 0, i*8 + 1, i*8 + 2);
      m_data[i].leftOffset  = i*8 + 3;
      m_data[i].boxMax      = float3(i*8 + 4, i*8 + 5,i*8 + 6);
      m_data[i].escapeIndex = i*8 + 7;
    }

    //for(uint32_t i=0;i < m_data.size();i++) {
    //  m_data[i].boxMin = float4(i*8 + 0, i*8 + 1, i*8 + 2, as_float(i*8 + 3));
    //  m_data[i].boxMax = float4(i*8 + 4, i*8 + 5, i*8 + 6, as_float(i*8 + 7));
    //}

    std::cout << "sizeof(float3)  = " << sizeof(float3) << std::endl;
    std::cout << "sizeof(BVHNode) = " << sizeof(BVHNode) << std::endl;
  }

  virtual void Test(float* a_data __attribute__((size("a_size*8"))), unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);

  std::vector<BVHNode> m_data;

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
