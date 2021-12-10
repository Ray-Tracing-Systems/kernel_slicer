#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"

#include <vector>
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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TestClass // : public DataClass
{
public:

  TestClass(int a_maxThreads = 1)
  {
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.01f, 1000.0f);
    m_worldViewProjInv  = inverse4x4(proj);
    InitSpheresScene(10);
    InitRandomGens(a_maxThreads);
  }

  void InitRandomGens(int a_maxThreads);

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY __attribute__((size("tidX", "tidY"))));
  void CastSingleRay(uint tid, const uint* in_pakedXY __attribute__((size("tid"))), 
                                     uint* out_color  __attribute__((size("tid"))));
  void StupidPathTrace(uint tid, const uint a_maxDepth, const uint* in_pakedXY __attribute__((size("tid"))), 
                                                            float4* out_color  __attribute__((size("tid"))));
  
  virtual void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passesNum);
  virtual void CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passesNum);
  virtual void StupidPathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passesNum);
  
  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                       Lite_Hit* out_hit);
  
  void kernel_GetMaterialColor(uint tid, const Lite_Hit* in_hit, 
                               uint* out_color);

  void kernel_InitAccumData(uint tid, float4* accumColor, float4* accumuThoroughput);
  
  void kernel_NextBounce(uint tid, const Lite_Hit* in_hit, 
                         float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput);

  void kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color);

  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, 
                                float4* out_color);

protected:

  void InitSpheresScene(int a_numSpheres, int a_seed = 0);

  float4x4                     m_worldViewProjInv;
  std::vector<float4>          spheresPosRadius;
  std::vector<SphereMaterial>  spheresMaterials;
  std::vector<RandomGen>       m_randomGens;
  std::vector<float>           m_unusedVector;

  float m_executionTimePT = 0.0f;
};

#endif