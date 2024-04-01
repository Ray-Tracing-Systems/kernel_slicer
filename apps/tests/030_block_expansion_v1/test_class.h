#pragma once
#include <cstdint>

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;


class TestClass
{
public:
  TestClass(){}

  virtual void Test(uint numElements, float*  out_buffer [[size("numElements")]]);
  
  template<int bsize> 
  void kernelBE1D_Test(uint blockNum, float* out_buffer); 

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
