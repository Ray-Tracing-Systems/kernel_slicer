#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

#include "CrossRT.h"

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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TestClass // : public DataClass
{
public:

  TestClass(int a_maxThreads = 1)
  {
    const float4x4 proj = perspectiveMatrix(45.0f, 1.0f, 0.01f, 100.0f);
    m_worldViewProjInv  = inverse4x4(proj);
    InitRandomGens(a_maxThreads);
    m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT("")); 
  }

  ~TestClass() {m_pAccelStruct = nullptr;}

  void InitRandomGens(int a_maxThreads);
  virtual int LoadScene(const char* meshPath); ///< Fix/Update this function to

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY __attribute__((size("tidX", "tidY"))) );
  void CastSingleRay(uint tid, const uint* in_pakedXY __attribute__((size("tid"))), 
                                     uint* out_color  __attribute__((size("tid"))) );
  
  virtual void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum);
  virtual void CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);
  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);      

  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                       Lite_Hit* out_hit, float4* out_surfPos);

  void kernel_CalcShadow(uint tid, const float4* in_pos, float* out_shadow);

  void kernel_GetTestColor(uint tid, const Lite_Hit* in_hit, const float* in_shadow, uint* out_color);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  float3    testColor = float3(0, 1, 1);
  uint32_t  m_emissiveMaterialId = 0;
  LightGeom m_lightGeom = {float3(-1.0f, 2.0f, -1.0f), 
                           float3(+1.0f, 2.0f, +1.0f)   
                           };

  static constexpr uint HIT_TRIANGLE_GEOM   = 0;
  static constexpr uint HIT_FLAT_LIGHT_GEOM = 1;

protected:

  float3 m_camPos = float3(0.0f, 0.85f, 4.5f);
  void InitSceneMaterials(int a_numSpheres, int a_seed = 0);

  std::vector<uint32_t>        m_materialIds;
  std::vector<float4>          m_materials;

  float4x4                     m_worldViewProjInv;
  std::vector<RandomGen>       m_randomGens;

  std::shared_ptr<ISceneObject> m_pAccelStruct = nullptr;
  float m_executionTimeCast = 0.0f;
};

#endif