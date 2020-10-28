#include <iostream>
#include <fstream>
#include <vector>

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL

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

uint fakeOffset (uint x, uint y) { return 0; } 

template<class T> struct MyTestVector
{
  T data[6];
};

class TestClass // : public DataClass
{
public:

  TestClass()
  {
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
    m_worldViewProjInv  = inverse4x4(proj);
  }
  
  void MainFunc(uint tidX, uint tidY, uint* out_color);

  void kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  void kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                       Lite_Hit* out_hit, uint tidX, uint tidY);
  
  void kernel_TestColor(const Lite_Hit* in_hit, uint* out_color, uint tidX, uint tidY);

private:

  float4x4 m_worldViewProjInv;
  float m_data1;
  float m_data2[4];
  std::vector<float>  m_someBufferData;
  MyTestVector<float> m_someBufferData2;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const float3 rayDir = EyeRayDir((float)tidX, (float)tidY, (float)WIN_WIDTH, (float)WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = make_float3(0.0f, 0.0f, 0.0f);
  
  rayPosAndNear[fakeOffset(tidX,tidY)] = to_float4(rayPos, 0.0f);
  rayDirAndFar [fakeOffset(tidX,tidY)] = to_float4(rayDir, MAXFLOAT);
  flags        [fakeOffset(tidX,tidY)] = 0;
}

void TestClass::kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                                Lite_Hit* out_hit, uint tidX, uint tidY)
{
  const Lite_Hit hit = RayTraceImpl(to_float3(rayPosAndNear[fakeOffset(tidX,tidY)]), to_float3(rayDirAndFar [fakeOffset(tidX,tidY)]));

  out_hit     [fakeOffset(tidX,tidY)]   = hit;
  rayDirAndFar[fakeOffset(tidX,tidY)].w = hit.t;
}

void TestClass::kernel_TestColor(const Lite_Hit* in_hit, uint* out_color, uint tidX, uint tidY)
{
  float x = 2.0f;
  if(in_hit[fakeOffset(tidX,tidY)].primId != -1)
    out_color[pitchOffset(tidX,tidY)] = 0x000000FF;
  else
    out_color[pitchOffset(tidX,tidY)] = 0x00FF0000;
}

void TestClass::MainFunc(uint tidX, uint tidY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;

  kernel_InitEyeRay(&flags, &rayPosAndNear, &rayDirAndFar, tidX, tidY);

  Lite_Hit hit;
  kernel_RayTrace(&rayPosAndNear, &rayDirAndFar, 
                  &hit, tidX, tidY);
  
  kernel_TestColor(&hit, 
                   out_color, tidX, tidY);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Bitmap.h"

void test_class_exec()
{
  TestClass test;
  std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);
  
  for(int y=0;y<WIN_HEIGHT;y++)
  {
    for(int x=0;x<WIN_WIDTH;x++)
      test.MainFunc(x,y,pixelData.data());
  }

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  //std::cout << resultingColor.x << " " << resultingColor.y << " " << resultingColor.z << std::endl;
  return;
}