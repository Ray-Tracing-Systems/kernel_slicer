#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "include/crandom.h"

#include <vector>
#include <iostream>
#include <fstream>

#include "include/bvh.h"

struct BxDFSample
{
  float3 brdfVal;
  float  pdfVal;
  float3 newDir;
  uint   flags;
};

struct IMaterial
{
  static constexpr uint32_t TAG_EMPTY      = 0;    // !!! #REQUIRED by kernel slicer: Empty/Default impl must have zero both tag and offset
  static constexpr uint32_t TAG_LAMBERT    = 1; 
  static constexpr uint32_t TAG_MIRROR     = 2; 
  static constexpr uint32_t TAG_EMISSIVE   = 3; 
  static constexpr uint32_t TAG_GGX_GLOSSY = 4;

  IMaterial(){}  // Dispatching on GPU hierarchy must not have destructors, especially virtual      

  virtual uint32_t   GetTag()   const  { return 0; };
  virtual float3     GetColor() const  { return float3(0.0f); };
  virtual BxDFSample SampleAndEvalBxDF(float4 rayPosAndNear, float4 rayDirAndFar, SurfaceHit hit, float2 uv) const  { BxDFSample res; return res; }

  float m_color[3];
  float roughness;
  uint32_t m_tag;
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
                       Lite_Hit* out_hit, float2* out_bars, uint* out_mid);
  
  void kernel_RealColorToUint32(uint tid, uint mid, uint* out_color);

  void kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, 
                                float4* out_color);
 
  void kernel_NextBounce(uint tid, uint mid, const Lite_Hit* in_hit, const float2* in_bars, 
                         const uint32_t* in_indices, const float4* in_vpos, const float4* in_vnorm,
                         float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen, 
                         float4* accumColor, float4* accumThoroughput);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

  //  BVHTree                  m_bvhTree;
  std::vector<struct BVHNode>  m_nodes;
  std::vector<struct Interval> m_intervals;
  std::vector<uint32_t>        m_indicesReordered;
  std::vector<uint32_t>        m_materialIds;
  std::vector<float4>          m_vPos4f;      // copy from m_mesh
  std::vector<float4>          m_vNorm4f;     // copy from m_mesh

  std::vector<IMaterial>       m_materials;

  virtual void Update_m_materials(size_t a_start, size_t a_size) {}

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
  LambertMaterial(float3 a_color) { m_color[0] = a_color[0]; m_color[1] = a_color[1]; m_color[2] = a_color[2]; m_tag = GetTag(); }
  ~LambertMaterial() = delete;  

  uint32_t GetTag()   const override { return TAG_LAMBERT; }      
  float3   GetColor() const override { return float3(m_color[0], m_color[1], m_color[2]); }
  
  BxDFSample SampleAndEvalBxDF(float4 rayPosAndNear, float4 rayDirAndFar, SurfaceHit hit, float2 uv) const override
  {
    const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hit.norm, hit.norm, 1.0f);
    const float  cosTheta = dot(newDir, hit.norm);

    BxDFSample res;
    res.pdfVal  = cosTheta * INV_PI;
    res.brdfVal = (cosTheta > 1e-5f) ? float3(m_color[0], m_color[1], m_color[2]) * INV_PI : float3(0,0,0);
    res.newDir  = newDir;
    res.flags   = 0;
    return res;
  }
 

};

struct PerfectMirrorMaterial : public IMaterial
{
  PerfectMirrorMaterial() { m_color[0] = 0.85f; m_color[1] = 0.85f; m_color[2] = 0.85f; m_tag = GetTag();}
  ~PerfectMirrorMaterial() = delete;

  uint32_t GetTag()   const override { return TAG_MIRROR; }
  float3   GetColor() const override { return float3(0,0,0); }
  
  BxDFSample SampleAndEvalBxDF(float4 rayPosAndNear, float4 rayDirAndFar, SurfaceHit hit, float2 uv) const override
  {
    const float3 ray_dir = to_float3(rayDirAndFar);
    float3 newDir = reflect(ray_dir, hit.norm);
    if (dot(ray_dir, hit.norm) > 0.0f)
      newDir = ray_dir;

    const float cosTheta = dot(newDir, hit.norm);

    BxDFSample res;
    res.pdfVal  = 1.0f;
    res.brdfVal = (cosTheta > 1e-5f) ? float3(m_color[0], m_color[1], m_color[2]) * (1.0f / std::max(cosTheta, 1e-5f)): float3(0,0,0);
    res.newDir  = newDir;
    res.flags   = 0;
    return res;
  }
};

struct EmissiveMaterial : public IMaterial
{
  EmissiveMaterial(float a_intensity) { roughness = a_intensity; m_tag = GetTag();}
  ~EmissiveMaterial() = delete;

  uint32_t GetTag() const override { return TAG_EMISSIVE; }
  float3 GetColor() const override { return float3(1,1,1); }
  
  BxDFSample SampleAndEvalBxDF(float4 rayPosAndNear, float4 rayDirAndFar, SurfaceHit hit, float2 uv) const override
  {
    const float3 ray_dir = to_float3(rayDirAndFar);    
    float3 emissiveColor = roughness*GetColor();
    if(ray_dir.y <= 0.0f)
      emissiveColor = float3(0,0,0);

    BxDFSample res;
    res.pdfVal  = 1.0f;
    res.brdfVal = emissiveColor;
    res.newDir  = ray_dir;
    res.flags   = 1;
    return res;
  }
};

struct GGXGlossyMaterial : public IMaterial
{
  GGXGlossyMaterial(float3 a_color) { m_color[0] = a_color[0]; m_color[1] = a_color[1]; m_color[2] = a_color[2]; roughness = 0.3f; }
  ~GGXGlossyMaterial() = delete;

  uint32_t GetTag()   const override { return TAG_GGX_GLOSSY; }
  float3   GetColor() const override { return float3(m_color[0], m_color[1], m_color[2]); }

  BxDFSample SampleAndEvalBxDF(float4 rayPosAndNear, float4 rayDirAndFar, SurfaceHit hit, float2 uv) const override
  {
    const float3 ray_dir = to_float3(rayDirAndFar);
    
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
    
      Pss    = D * G / std::max(4.0f * dotNV * dotNL, 1e-6f);        
      outPdf = D * dotNH / std::max(4.0f * dotHV, 1e-6f);
    }  

    BxDFSample res;
    res.pdfVal  = outPdf;
    res.brdfVal = float3(m_color[0], m_color[1], m_color[2]) * Pss;
    res.newDir  = newDir;
    res.flags   = 0;
    return res;
  }

};

struct EmptyMaterial : public IMaterial
{
  EmptyMaterial() { m_tag = GetTag();}
  ~EmptyMaterial() = delete;

  uint32_t GetTag()   const override { return TAG_EMPTY; }
};

#endif