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
    InitRandomGens(a_maxThreads);
    m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT(""), [](ISceneObject *p) { DeleteSceneRT(p); } ); 
  }

  ~TestClass() {m_pAccelStruct = nullptr; }

  void InitRandomGens(int a_maxThreads);
  virtual int LoadScene(const char* bvhPath);

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY  __attribute__((size("tidX", "tidY"))) );
  void CastSingleRay (uint tid, const uint* in_pakedXY __attribute__((size("tid"))), 
                                      uint* out_color  __attribute__((size("tid"))) );
  void NaivePathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY __attribute__((size("tid"))), 
                                                     float4* out_color  __attribute__((size("tid"))) );

  virtual void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum);
  virtual void CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum);
  virtual void NaivePathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class
  virtual void SceneRestrictions(uint32_t a_restrictions[4]) const       // will be used by RTX code
  {
    uint32_t maxMeshes            = 1024;
    uint32_t maxTotalVertices     = 4'000'000;
    uint32_t maxTotalPrimitives   = 4'000'000;
    uint32_t maxPrimitivesPerMesh = 1'000'000;

    a_restrictions[0] = maxMeshes;
    a_restrictions[1] = maxTotalVertices;
    a_restrictions[2] = maxTotalPrimitives;
    a_restrictions[3] = maxPrimitivesPerMesh;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
  void kernel_InitEyeRay2(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumuThoroughput);        

  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                       Lite_Hit* out_hit, float2* out_bars);

  void kernel_GetRayColor(uint tid, const Lite_Hit* in_hit, uint* out_color);

  void kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                         float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput);
  
  void kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color);

  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, 
                                float4* out_color);

protected:

  float3 m_camPos = float3(0.0f, 0.85f, 4.5f);
  void InitSceneMaterials(int a_numSpheres, int a_seed = 0);

  std::vector<float4>          m_materials;
  std::vector<uint32_t>        m_matIdOffsets;  ///< offset = m_matIdOffsets[geomId]
  std::vector<uint32_t>        m_matIdByPrimId; ///< matId  = m_matIdByPrimId[offset + primId]
  std::vector<uint32_t>        m_triIndices;    ///< (A,B,C) = m_triIndices[(offset + primId)*3 + 0/1/2]

  std::vector<uint32_t>        m_vertOffset;    ///< vertOffs = m_vertOffset[geomId]
  std::vector<float4>          m_vPos4f;        ///< vertPos  = m_vPos4f [vertOffs + vertId]
  std::vector<float4>          m_vNorm4f;       ///< vertNorm = m_vNorm4f[vertOffs + vertId]

  float4x4                     m_projInv;
  float4x4                     m_worldViewInv;
  std::vector<RandomGen>       m_randomGens;
  std::vector<float4x4>        m_normMatrices; ///< per instance normal matrix, local to world

  std::shared_ptr<ISceneObject> m_pAccelStruct = nullptr;
  float m_executionTimePT = 0.0f;
};

#endif