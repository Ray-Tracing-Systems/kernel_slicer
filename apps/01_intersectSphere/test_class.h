#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL

#include <iostream>
#include <fstream>

template<class T> struct MyTestVector
{
  T _data[6];
  T* data() { return &_data[0]; }
};

class TestClass // : public DataClass
{
public:

  TestClass()
  {
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
    m_worldViewProjInv  = inverse4x4(proj);
    m_data1 = 0;
    m_data2[0] = 1.0f;
    m_data2[1] = 2.0f;
    m_data2[2] = 0.5f;
  }
  
  void MainFunc(uint tidX, uint tidY, uint* out_color __attribute__((size("tidX", "tidY"))));
  virtual void MainFuncBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses = 1);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                       Lite_Hit* out_hit, uint tidX, uint tidY);
  
  void kernel_TestColor(const Lite_Hit* in_hit, uint* out_color, uint tidX, uint tidY, uint sphereColor, uint backColor);

protected:

  float4x4 m_worldViewProjInv;
  unsigned int m_data1;
  float m_data2[3];
  std::vector<unsigned int>  m_someBufferData;
  MyTestVector<unsigned int> m_someBufferData2;
};

#endif