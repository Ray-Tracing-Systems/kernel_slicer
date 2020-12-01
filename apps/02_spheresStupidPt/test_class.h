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

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum MATERIAL_FLAGS{ MTL_EMISSIVE = 1 };

typedef struct MaterialT
{
  float3   color;
  uint32_t flags;

} SphereMaterial;

static inline bool   IsMtlEmissive(const SphereMaterial* a_mtl)       { return (a_mtl->flags & MTL_EMISSIVE) != 0; }
static inline float3 GetMtlDiffuseColor(const SphereMaterial* a_mtl)  { return IsMtlEmissive(a_mtl) ? float3(0,0,0) : a_mtl->color; }
static inline float3 GetMtlEmissiveColor(const SphereMaterial* a_mtl) { return IsMtlEmissive(a_mtl) != 0 ? a_mtl->color : float3(0,0,0); }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TestClass // : public DataClass
{
public:

  TestClass()
  {
    const float4x4 proj = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
    m_worldViewProjInv  = inverse4x4(proj);
    InitSpheresScene(10);
  }

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void CastSingleRay(uint tid, uint* in_pakedXY, uint* out_color);

  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void kernel_InitEyeRay(uint tid, const uint* packedXY, uint* flags, float4* rayPosAndNear, float4* rayDirAndFar);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!

  void kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar, 
                       Lite_Hit* out_hit);
  
  void kernel_GetMaterialColor(uint tid, const Lite_Hit* in_hit, 
                               uint* out_color);

protected:

  void InitSpheresScene(int a_numSpheres, int a_seed = 0);

  float4x4                     m_worldViewProjInv;
  std::vector<float4>          spheresPosRadius;
  std::vector<SphereMaterial>  spheresMaterials;
};

#endif