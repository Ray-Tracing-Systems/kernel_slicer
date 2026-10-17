#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <memory>

#include "CrossRT.h"

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

  virtual void PackXY(uint tidX, uint tidY, uint* out_pakedXY __attribute__((size("tidX", "tidY"))) );
  virtual void CastSingleRay(uint tid, const uint* in_pakedXY __attribute__((size("tid"))),
                                             uint* out_color  __attribute__((size("tid"))) );

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  virtual void kernel2D_PackXY(uint tidX, uint tidY, uint* out_pakedXY);
  virtual void kernel1D_CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color); ///< all computations for a ray in one kernel

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // ordinary member functions called from 'kernel1D_CastSingleRay'
  //
  Lite_Hit RayTrace(float4 rayPosAndNear, float4 rayDirAndFar);
  float4   ShadowRayOrigin(float4 rayPosAndNear, float4 rayDirAndFar, float hitT);
  float    CalcShadow(uint tid, float4 surfPos);
  uint     GetTestColor(Lite_Hit hit, float shadow);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  float3    testColor = float3(0, 1, 1);
  uint32_t  m_emissiveMaterialId = 0;
  LightGeom m_lightGeom = {float4(-1.0f, 2.0f, -1.0f, 1.0f), 
                           float4(+1.0f, 2.0f, +1.0f, 1.0f)   
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