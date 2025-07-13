#pragma once
#include <cstdint>

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::float2;
using LiteMath::float4;
using LiteMath::float4x4;

class TestClass
{
public:
  TestClass() { }

  virtual void Test(float* a_data [[size("a_size")]], int a_size);
  void kernel1D_Test(float* a_data, int a_size);

  float fSumm(int tid);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
