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

struct MyFloat3 { float data[3]; };

struct BadStruct
{
  float2 xy;
  float  depth;
};

class Padding
{
public:
  Padding()
  { 
    m_data1 = {{1.0f,2.0f,3.0f}, {4.0f,5.0f,6.0f}, {7.0f,8.0f,9.0f}};
    m_data2 = {{-1.0f,-2.0f,-3.0f}, {-4.0f,-5.0f,-6.0f}, {-7.0f,-8.0f,-9.0f}}; 
    m_data3 = { {{0.0f,0.0f},0.0f}, {{0.0f,0.0f},0.0f}, {{0.0f,0.0f},0.0f}, {{0.0f,0.0f},0.0f} };
  }

  virtual void Test(float* a_data __attribute__((size("a_size"))), unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);

  std::vector<float3>   m_data1;
  std::vector<MyFloat3> m_data2;
  std::vector<BadStruct> m_data3;

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
