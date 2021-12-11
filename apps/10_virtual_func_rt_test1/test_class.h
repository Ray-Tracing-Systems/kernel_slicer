#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"

#include <vector>
#include <iostream>
#include <fstream>

#include "include/bvh.h"

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

class TestClass;

struct IMaterial
{
  static constexpr uint32_t TAG_BITS = 4;          // number bits for type encoding in index; 
  static constexpr uint32_t TAG_MASK = 0xF0000000; // mask which we can get from TAG_BITS
  static constexpr uint32_t OFS_MASK = 0x0FFFFFFF; // (32 - TAG_BITS) is left for object/thread id.

  static constexpr uint32_t TAG_EMPTY      = 0;    // !!! #REQUIRED by kernel slicer: Empty/Default impl must have zero both tag and offset
  static constexpr uint32_t TAG_LAMBERT    = 1; 
  static constexpr uint32_t TAG_MIRROR     = 2; 
  static constexpr uint32_t TAG_EMISSIVE   = 3; 
  static constexpr uint32_t TAG_GGX_GLOSSY = 4;

  IMaterial(){}  // Dispatching on GPU hierarchy must not have destructors, especially virtual      

  virtual uint32_t GetTag() const = 0;
  virtual size_t   GetSizeOf() const = 0;

  virtual void   kernel_GetColor(uint tid, __global uint* out_color, const TestClass* a_pGlobals) const = 0;

  virtual void   kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                                   const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                                   float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen, 
                                   float4* accumColor, float4* accumThoroughput) const = 0;
};


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
  }

  void InitRandomGens(int a_maxThreads);
  int LoadScene(const char* bvhPath, const char* meshPath);
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  void PackXY(uint tidX, uint tidY, uint* out_pakedXY __attribute__((size("tidX", "tidY"))) );
  void CastSingleRay(uint tid, const uint* in_pakedXY __attribute__((size("tid"))), 
                                     uint* out_color  __attribute__((size("tid"))) );
  void NaivePathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY __attribute__((size("tid"))), 
                                                     float4* out_color  __attribute__((size("tid"))) );

  virtual void PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum);
  virtual void CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum);
  virtual void NaivePathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum);

  virtual void CommitDeviceData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]); // will be overriden in generated class

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  void kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY);

  void kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar);        // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
  void kernel_InitEyeRay2(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumuThoroughput);        

  bool kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                       Lite_Hit* out_hit, float2* out_bars);
  
  void kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color);

  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, 
                                float4* out_color);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  IMaterial* kernel_MakeMaterial(uint tid, const Lite_Hit* in_hit);

  float3    testColor = float3(0, 1, 1);
  uint32_t  m_emissiveMaterialId = 0;
  LightGeom m_lightGeom = {float3(-0.3f, 2.0f, -0.3f), 
                           float3(+0.3f, 2.0f, +0.3f)   
                           };

  static constexpr uint HIT_TRIANGLE_GEOM   = 0;
  static constexpr uint HIT_FLAT_LIGHT_GEOM = 1;

protected:
  float3 camPos = float3(0.0f, 0.85f, 4.5f);
  void InitSceneMaterials(int a_numSpheres, int a_seed = 0);

  uint32_t PackObject(uint32_t*& pData, IMaterial* a_pObject);

//  BVHTree                      m_bvhTree;
  std::vector<struct BVHNode>  m_nodes;
  std::vector<struct Interval> m_intervals;
  std::vector<uint32_t>        m_indicesReordered;
  std::vector<uint32_t>        m_materialIds;
  std::vector<float4>          m_vPos4f;      // copy from m_mesh
  std::vector<float4>          m_vNorm4f;     // copy from m_mesh

  std::vector<uint32_t>        m_materialData;
  std::vector<uint32_t>        m_materialOffsets;

  float4x4                     m_worldViewProjInv;
  std::vector<RandomGen>       m_randomGens;

  float m_executionTimePT = 0.0f;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct LambertMaterial : public IMaterial
{
  LambertMaterial(float3 a_color) { m_color[0] = a_color[0]; m_color[1] = a_color[1]; m_color[2] = a_color[2]; }
  ~LambertMaterial() = delete;  

  uint32_t GetTag()    const override { return TAG_LAMBERT; }
  size_t   GetSizeOf() const override { return sizeof(LambertMaterial); }                  

  float m_color[3];

  void  kernel_GetColor(uint tid, uint* out_color, const TestClass* a_pGlobals) const override 
  { 
    out_color[tid] = RealColorToUint32_f3(float3(m_color[0], m_color[1], m_color[2])); 
  }

  void   kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                           const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                           float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen, 
                           float4* accumColor, float4* accumThoroughput) const override
  {
    const Lite_Hit lHit  = *in_hit;
    const float3 ray_dir = to_float3(*rayDirAndFar);
    
    SurfaceHit hit;
    hit.pos  = to_float3(*rayPosAndNear) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, *in_bars, in_indices, in_vnorm);
  
    RandomGen gen   = pGen[tid];
    const float2 uv = rndFloat2_Pseudo(&gen);
    pGen[tid]       = gen;
  
    const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hit.norm, hit.norm, 1.0f);
    const float  cosTheta = dot(newDir, hit.norm);
  
    const float pdfVal   = cosTheta * INV_PI;
    const float3 brdfVal = (cosTheta > 1e-5f) ? float3(m_color[0], m_color[1], m_color[2]) * INV_PI : float3(0,0,0);
    const float3 bxdfVal = brdfVal * (1.0f / fmax(pdfVal, 1e-10f));
    
    *rayPosAndNear    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
    *rayDirAndFar     = to_float4(newDir, MAXFLOAT);
    *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f);
  }
 

};

struct PerfectMirrorMaterial : public IMaterial
{
  ~PerfectMirrorMaterial() = delete;

  uint32_t GetTag()    const override { return TAG_MIRROR; }
  size_t   GetSizeOf() const override { return sizeof(PerfectMirrorMaterial); }

  void kernel_GetColor(uint tid, uint* out_color, const TestClass* a_pGlobals) const override 
  { 
    out_color[tid] = RealColorToUint32_f3(a_pGlobals->testColor); 
  }
  
  void   kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                           const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                           float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen,
                           float4* accumColor, float4* accumThoroughput) const override
  {
    const Lite_Hit lHit  = *in_hit;
    const float3 ray_dir = to_float3(*rayDirAndFar);
  
    SurfaceHit hit;
    hit.pos  = to_float3(*rayPosAndNear) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, *in_bars, in_indices, in_vnorm);
     
    float3 newDir = reflect(ray_dir, hit.norm);
    if (dot(ray_dir, hit.norm) > 0.0f)
      newDir = ray_dir;
    
    *rayPosAndNear     = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
    *rayDirAndFar      = to_float4(newDir, MAXFLOAT);
    *accumThoroughput *= float4(0.85f, 0.85f, 0.85f, 0.0f);
  }
};

struct EmissiveMaterial : public IMaterial
{
  EmissiveMaterial(float a_intensity) : intensity(a_intensity) {}
  ~EmissiveMaterial() = delete;

  uint32_t GetTag()    const override { return TAG_EMISSIVE; }
  size_t   GetSizeOf() const override { return sizeof(EmissiveMaterial); }
  
  float3 GetColor() const { return float3(1,1,1); }
  
  void   kernel_GetColor(uint tid, uint* out_color, const TestClass* a_pGlobals) const override 
  { 
    out_color[tid] = RealColorToUint32_f3(intensity*GetColor()); 
  }

  void   kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                           const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                           float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen,
                           float4* accumColor, float4* accumThoroughput) const override
  {
    float4 emissiveColor = to_float4(intensity*GetColor(), 0.0f);
    const float3 ray_dir = to_float3(*rayDirAndFar);
    if(ray_dir.y <= 0.0f)
      emissiveColor = float4(0,0,0,0);
    
    *accumColor    = emissiveColor*(*accumThoroughput);
    *rayPosAndNear = make_float4(0,10000000.0f,0,0); // shoot ray out of scene
    *rayDirAndFar  = make_float4(0,1.0f,0,0);        // 
  }

  float  intensity;
};

struct GGXGlossyMaterial : public IMaterial
{
  GGXGlossyMaterial(float3 a_color) { color[0] = a_color[0]; color[1] = a_color[1]; color[2] = a_color[2]; roughness = 0.3f; }
  ~GGXGlossyMaterial() = delete;

  uint32_t GetTag()    const override { return TAG_GGX_GLOSSY; }
  size_t   GetSizeOf() const override { return sizeof(GGXGlossyMaterial); }
  
  void  kernel_GetColor(uint tid, uint* out_color, const TestClass* a_pGlobals) const override 
  { 
    float redColor = std::max(1.0f, color[0]);
    out_color[tid] = RealColorToUint32_f3(float3(redColor, color[1], color[2])); 
  }

  void   kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                           const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                           float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen,
                           float4* accumColor, float4* accumThoroughput) const override
  {
    const Lite_Hit lHit  = *in_hit;
    const float3 ray_dir = to_float3(*rayDirAndFar);
  
    SurfaceHit hit;
    hit.pos  = to_float3(*rayPosAndNear) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, *in_bars, in_indices, in_vnorm);
  
    RandomGen gen   = pGen[tid];
    const float2 uv = rndFloat2_Pseudo(&gen);
    pGen[tid]       = gen;
  
    const float  roughSqr   = roughness * roughness;
    float3 nx, ny, nz       = hit.norm;
    CoordinateSystem(nz, &nx, &ny);
  
    // to PBRT coordinate system
    const float3 wo = make_float3(-dot(ray_dir, nx), -dot(ray_dir, ny), -dot(ray_dir, nz));  // wo (output) = v = ray_dir
  
    // Compute sampled half-angle vector wh
    const float phi       = uv.x * M_TWOPI;
    const float cosTheta  = clamp(sqrt((1.0f - uv.y) / (1.0f + roughSqr * roughSqr * uv.y - uv.y)), 0.0f, 1.0f);
    const float sinTheta  = sqrt(1.0f - cosTheta * cosTheta);
    const float3 wh       = SphericalDirectionPBRT(sinTheta, cosTheta, phi);
    const float3 wi       = (2.0f * dot(wo, wh) * wh) - wo;                  // Compute incident direction by reflecting about wh. wi (input) = light
    const float3 newDir   = normalize(wi.x*nx + wi.y*ny + wi.z*nz);          // back to normal coordinate system
  
    float Pss         = 1.0f;  // Pass single-scattering  
    const float3 v    = ray_dir * (-1.0f);
    const float3 l    = newDir;
    const float dotNV = dot(hit.norm, v);
    const float dotNL = dot(hit.norm, l);
    
    float outPdf = 1.0f; 
    if (dotNV < 1e-6f || dotNL < 1e-6f)
    {
      Pss    = 0.0f;
      outPdf = 1.0f;
    }
    else
    {
      const float3 h    = normalize(v + l);  // half vector.
      const float dotNV = dot(hit.norm, v);
      const float dotNH = dot(hit.norm, h);
      const float dotHV = dot(h, v);
    
      // Fresnel is not needed here, because it is used for the blend with diffusion.
      const float D = GGX_Distribution(dotNH, roughSqr);
      const float G = GGX_GeomShadMask(dotNV, roughSqr) * GGX_GeomShadMask(dotNL, roughSqr); 
    
      Pss    = D * G / fmax(4.0f * dotNV * dotNL, 1e-6f);        
      outPdf = D * dotNH / fmax(4.0f * dotHV, 1e-6f);
    }  
  
    *rayPosAndNear    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
    *rayDirAndFar     = to_float4(newDir, MAXFLOAT);
    *accumThoroughput *= cosTheta*to_float4(float3(color[0], color[1], color[2]) * Pss * (1.0f/fmax(outPdf, 1e-5f)), 0.0f);
  }

  float color[3];
  float roughness;
};

struct EmptyMaterial : public IMaterial
{
  EmptyMaterial() {}
  ~EmptyMaterial() = delete;

  uint32_t GetTag() const override { return TAG_EMPTY; }
  size_t   GetSizeOf() const override { return sizeof(EmptyMaterial); }

  void kernel_GetColor(uint tid, uint* out_color, const TestClass* a_pGlobals) const override  { }

  void   kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                           const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                           float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen,
                           float4* accumColor, float4* accumThoroughput) const override { }
};



#endif