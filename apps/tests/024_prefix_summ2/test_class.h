#pragma once
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>

#define LAYOUT_STD140
#include "LiteMath.h"

using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;

class PrefSummTest
{
public:
  PrefSummTest(){}
  
  virtual void Resize(size_t n) { m_internalData.resize(n); }

  virtual void PrefixSumm(const int* a_data  __attribute__((size("a_size"))), unsigned int a_size,
                          int* a_outExc __attribute__((size("a_size"))));

  virtual void kernel1D_CopyData(const int* a_data, unsigned int a_size);                        

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  std::vector<uint32_t> m_internalData;
};
