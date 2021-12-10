#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL

#include <iostream>
#include <fstream>

static inline float4x4 perspectiveMatrix(float fovy, float aspect, float zNear, float zFar)
{
  const float ymax = zNear * tanf(fovy * 3.14159265358979323846f / 360.0f);
  const float xmax = ymax * aspect;
  const float left = -xmax;
  const float right = +xmax;
  const float bottom = -ymax;
  const float top = +ymax;
  const float temp = 2.0f * zNear;
  const float temp2 = right - left;
  const float temp3 = top - bottom;
  const float temp4 = zFar - zNear;
  float4x4 res;
  res.m_col[0] = float4{ temp / temp2, 0.0f, 0.0f, 0.0f };
  res.m_col[1] = float4{ 0.0f, temp / temp3, 0.0f, 0.0f };
  res.m_col[2] = float4{ (right + left) / temp2,  (top + bottom) / temp3, (-zFar - zNear) / temp4, -1.0 };
  res.m_col[3] = float4{ 0.0f, 0.0f, (-temp * zFar) / temp4, 0.0f };
  return res;
}

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