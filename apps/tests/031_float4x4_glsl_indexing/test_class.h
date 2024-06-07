#pragma once
#include <cstdint>

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::float4;
using LiteMath::float4x4;

class TestClass
{
public:
  TestClass()
  {
    m_testMatrix = float4x4(1.0f, 2.0f, 3.0f, 4.0f, 
                            5.0f, 6.0f, 7.0f, 8.0f,
                            9.0f, 10.0f, 11.0f, 12.0f,
                            13.0f, 14.0f, 15.0f, 16.0f);

    m_testMatrices.resize(2);
    for(size_t i=0;i<m_testMatrices.size();i++) {
      m_testMatrices[i][0][0] = 1.0f;
      m_testMatrices[i][0][1] = 2.0f;
      m_testMatrices[i][0][2] = 3.0f;
      m_testMatrices[i][0][3] = 4.0f;

      m_testMatrices[i][1][0] = 5.0f;
      m_testMatrices[i][1][1] = 6.0f;
      m_testMatrices[i][1][2] = 7.0f;
      m_testMatrices[i][1][3] = 8.0f;

      m_testMatrices[i][2][0] = 9.0f;
      m_testMatrices[i][2][1] = 10.0f;
      m_testMatrices[i][2][2] = 11.0f;
      m_testMatrices[i][2][3] = 12.0f;

      m_testMatrices[i][3][0] = 13.0f;
      m_testMatrices[i][3][1] = 14.0f;
      m_testMatrices[i][3][2] = 15.0f;
      m_testMatrices[i][3][3] = 16.0f;
    }
      
  }

  virtual void Test(float4* a_data [[size("a_size")]], unsigned int a_size);
  void kernel1D_Test(float4* a_data, unsigned int a_size);

  std::vector<float4x4> m_testMatrices;
  float4x4              m_testMatrix;

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
