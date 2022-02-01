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

struct MyFloat5
{
  float data[5];
};

class Padding
{
public:
  Padding(){}

  virtual void Test(float* a_data __attribute__((size("a_size"))), unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);
  
  float4x4 m_testMat; // identity
  float4 m_data1 = float4(0,1,2,3);
  float2 m_data2 = float2(4,5);
  float  m_data3 = 6.0f;
  float  m_data4 = 7.0f;
  float  m_data5 = 8.0f;

  const MyFloat5 m_badData = {-1.0f,-2.0f,-3.0f,-4.0f,-5.0f};

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
