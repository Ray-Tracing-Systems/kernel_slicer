#pragma once
#include <vector>
#include <iostream>
#include <fstream>

#define LAYOUT_STD140
#include "LiteMath.h"

using LiteMath::float4x4;
using LiteMath::float4;
using LiteMath::float3;
using LiteMath::float2;

struct RectLightSource
{
  float4 intensity;
  float  value;
};

class Padding
{
public:
  Padding(){}

  virtual void Test(float* a_data __attribute__((size("a_size"))), unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);
  
  float4x4 matrix;
  float test1 = 5.0f;
  float test2 = 6.0f;
  float test3 = 7.0f;
   
  const RectLightSource m_badData = {{1.0f,2.0f,3.0f,4.0f}, 5.0f};

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
