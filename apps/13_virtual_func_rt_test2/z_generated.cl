/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
#include "include/OpenCLMath.h"
#include "include/bvh.h"
#include "include/crandom.h"
#include "include/BasicLogic.h"

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
__constant static uint HIT_TRIANGLE_GEOM = 0;
__constant static uint HIT_LIGHT_GEOM = 1;

#include "include/TestClass_ubo.h"

/////////////////////////////////////////////////////////////////////
/////////////////// declarations of IMaterial  
/////////////////////////////////////////////////////////////////////
__constant static uint32_t IMaterial_TAG_BITS = 4;
__constant static uint32_t IMaterial_TAG_MASK = 0xF0000000;
__constant static uint32_t IMaterial_OFS_MASK = 0x0FFFFFFF;
__constant static uint32_t IMaterial_TAG_EMPTY = 0;
__constant static uint32_t IMaterial_TAG_LAMBERT = 1;
__constant static uint32_t IMaterial_TAG_MIRROR = 2;
__constant static uint32_t IMaterial_TAG_EMISSIVE = 3;
__constant static uint32_t IMaterial_TAG_GGX_GLOSSY = 4;
__constant static uint32_t IMaterial_TAG_LAMBERT_MIX = 5;

  typedef struct LambertMaterialT 
  {
    float m_color[3];
  }LambertMaterial;  

void LambertMaterial_kernel_GetColor(
  __global const LambertMaterial* self, 
  uint tid, 
  __global uint* out_color, 
  __global const struct TestClass_UBO_Data* a_pGlobals)
  { 
    out_color[tid] = RealColorToUint32_f3(make_float3(self->m_color[0],self->m_color[1],self->m_color[2])); 
  }

void LambertMaterial_kernel_NextBounce(
  __global const LambertMaterial* self, 
  uint tid, 
  __global const Lite_Hit* in_hit, 
  __global const float2* in_bars, 
  __global const uint32_t* in_indices, 
  __global const float4* in_vpos, 
  __global const float4* in_vnorm, 
  __global float4* rayPosAndNear, 
  __global float4* rayDirAndFar, 
  __global RandomGen* pGen, 
  __global float4* accumColor, 
  __global float4* accumThoroughput)
  {
    const Lite_Hit lHit  = in_hit[tid];
    const float3 ray_dir = to_float3(rayDirAndFar[tid]);
    
    SurfaceHit hit;
    hit.pos  = to_float3(rayPosAndNear[tid]) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, in_bars[tid], in_indices, in_vnorm);
  
    RandomGen gen   = pGen[tid];
    const float2 uv = rndFloat2_Pseudo(&gen);
    pGen[tid]       = gen;
  
    const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hit.norm, hit.norm, 1.0f);
    const float  cosTheta = dot(newDir, hit.norm);
  
    const float pdfVal   = cosTheta * INV_PI;
    const float3 brdfVal = (cosTheta > 1e-5f) ? make_float3(self->m_color[0],self->m_color[1],self->m_color[2]) * INV_PI : make_float3(0,0,0);
    const float3 bxdfVal = brdfVal * (1.0f / fmax(pdfVal, 1e-10f));
    
    rayPosAndNear[tid]    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
    rayDirAndFar[tid]     = to_float4(newDir, MAXFLOAT);
    accumThoroughput[tid] *= cosTheta*to_float4(bxdfVal, 0.0f);
  }

  typedef struct PerfectMirrorMaterialT 
  {
  }PerfectMirrorMaterial;  

void PerfectMirrorMaterial_kernel_GetColor(
  __global const PerfectMirrorMaterial* self, 
  uint tid, 
  __global uint* out_color, 
  __global const struct TestClass_UBO_Data* a_pGlobals)
  { 
    out_color[tid] = RealColorToUint32_f3(a_pGlobals->testColor); 
  }

void PerfectMirrorMaterial_kernel_NextBounce(
  __global const PerfectMirrorMaterial* self, 
  uint tid, 
  __global const Lite_Hit* in_hit, 
  __global const float2* in_bars, 
  __global const uint32_t* in_indices, 
  __global const float4* in_vpos, 
  __global const float4* in_vnorm, 
  __global float4* rayPosAndNear, 
  __global float4* rayDirAndFar, 
  __global RandomGen* pGen, 
  __global float4* accumColor, 
  __global float4* accumThoroughput)
  {
    const Lite_Hit lHit  = in_hit[tid];
    const float3 ray_dir = to_float3(rayDirAndFar[tid]);
  
    SurfaceHit hit;
    hit.pos  = to_float3(rayPosAndNear[tid]) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, in_bars[tid], in_indices, in_vnorm);
     
    float3 newDir = reflect(ray_dir, hit.norm);
    if (dot(ray_dir, hit.norm) > 0.0f)
      newDir = ray_dir;
    
    rayPosAndNear[tid]     = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
    rayDirAndFar[tid]      = to_float4(newDir, MAXFLOAT);
    accumThoroughput[tid] *= make_float4(0.85f,0.85f,0.85f,0.0f);
  }

  typedef struct EmissiveMaterialT 
  {
    float  intensity;
  }EmissiveMaterial;  

float3 EmissiveMaterial_GetColor(
  __global const EmissiveMaterial* self)
  { return make_float3(1,1,1); }

void EmissiveMaterial_kernel_GetColor(
  __global const EmissiveMaterial* self, 
  uint tid, 
  __global uint* out_color, 
  __global const struct TestClass_UBO_Data* a_pGlobals)
  { 
    out_color[tid] = RealColorToUint32_f3(self->intensity*EmissiveMaterial_GetColor(self)); 
  }

void EmissiveMaterial_kernel_NextBounce(
  __global const EmissiveMaterial* self, 
  uint tid, 
  __global const Lite_Hit* in_hit, 
  __global const float2* in_bars, 
  __global const uint32_t* in_indices, 
  __global const float4* in_vpos, 
  __global const float4* in_vnorm, 
  __global float4* rayPosAndNear, 
  __global float4* rayDirAndFar, 
  __global RandomGen* pGen, 
  __global float4* accumColor, 
  __global float4* accumThoroughput)
  {
    const Lite_Hit lHit  = in_hit[tid];
    const float3 ray_dir = to_float3(rayDirAndFar[tid]);
    const float3 hitPos  = to_float3(rayPosAndNear[tid]) + lHit.t*ray_dir;

    float4 emissiveColor = BlueWhiteColor(hitPos*10.0f); // to_float4(intensity*GetColor(), 0.0f)

    
    accumColor[tid]    = emissiveColor*(accumThoroughput[tid]);
    rayPosAndNear[tid] = make_float4(0,10000000.0f,0,0); // shoot ray out of scene
    rayDirAndFar[tid]  = make_float4(0,1.0f,0,0);        // 
  }

  typedef struct GGXGlossyMaterialT 
  {
    float color[3];
    float roughness;
  }GGXGlossyMaterial;  

void GGXGlossyMaterial_kernel_GetColor(
  __global const GGXGlossyMaterial* self, 
  uint tid, 
  __global uint* out_color, 
  __global const struct TestClass_UBO_Data* a_pGlobals)
  { 
    float redColor = fmax(1.0f,self->color[0]);
    out_color[tid] = RealColorToUint32_f3(make_float3(redColor,self->color[1],self->color[2])); 
  }

void GGXGlossyMaterial_kernel_NextBounce(
  __global const GGXGlossyMaterial* self, 
  uint tid, 
  __global const Lite_Hit* in_hit, 
  __global const float2* in_bars, 
  __global const uint32_t* in_indices, 
  __global const float4* in_vpos, 
  __global const float4* in_vnorm, 
  __global float4* rayPosAndNear, 
  __global float4* rayDirAndFar, 
  __global RandomGen* pGen, 
  __global float4* accumColor, 
  __global float4* accumThoroughput)
  {
    const Lite_Hit lHit  = in_hit[tid];
    const float3 ray_dir = to_float3(rayDirAndFar[tid]);
  
    SurfaceHit hit;
    hit.pos  = to_float3(rayPosAndNear[tid]) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, in_bars[tid], in_indices, in_vnorm);
  
    RandomGen gen   = pGen[tid];
    const float2 uv = rndFloat2_Pseudo(&gen);
    pGen[tid]       = gen;
  
    const float  roughSqr   = self->roughness * self->roughness;
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
  
    rayPosAndNear[tid]    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
    rayDirAndFar[tid]     = to_float4(newDir, MAXFLOAT);
    accumThoroughput[tid] *= cosTheta*to_float4(make_float3(self->color[0],self->color[1],self->color[2]) * Pss * (1.0f/fmax(outPdf, 1e-5f)), 0.0f);
  }

  typedef struct LambertMaterialMixT 
  {
    float m_color[3];
  }LambertMaterialMix;  

void LambertMaterialMix_kernel_GetColor(
  __global const LambertMaterialMix* self, 
  uint tid, 
  __global uint* out_color, 
  __global const struct TestClass_UBO_Data* a_pGlobals)
  { 
    out_color[tid] = RealColorToUint32_f3(make_float3(self->m_color[0],self->m_color[1],self->m_color[2])); 
  }

void LambertMaterialMix_kernel_NextBounce(
  __global const LambertMaterialMix* self, 
  uint tid, 
  __global const Lite_Hit* in_hit, 
  __global const float2* in_bars, 
  __global const uint32_t* in_indices, 
  __global const float4* in_vpos, 
  __global const float4* in_vnorm, 
  __global float4* rayPosAndNear, 
  __global float4* rayDirAndFar, 
  __global RandomGen* pGen, 
  __global float4* accumColor, 
  __global float4* accumThoroughput)
  {
    const Lite_Hit lHit  = in_hit[tid];
    const float3 ray_dir = to_float3(rayDirAndFar[tid]);
    
    SurfaceHit hit;
    hit.pos  = to_float3(rayPosAndNear[tid]) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, in_bars[tid], in_indices, in_vnorm);
    
    const float3 pos   = hit.pos*8.0f;
    const float select = ov(make_float2(pos.z, pos.x), pos.y);

    RandomGen gen   = pGen[tid];
    const float4 uv = rndFloat4_Pseudo(&gen);
    pGen[tid]       = gen;

    float3 colorA = WhiteNoise(hit.pos*5.0f);
    float3 colorC = make_float3(0.8f,0.1f,0.8f);

    colorC = clamp(colorC, 0.0f, 1.0f);
    colorA = clamp(colorA, 0.0f, 1.0f);
   
    if(uv.z <= select)
    {
      const float  roughSqr   = 0.25f * 0.25f;
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
    
      rayPosAndNear[tid]    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
      rayDirAndFar[tid]     = to_float4(newDir, MAXFLOAT);
      accumThoroughput[tid] *= to_float4(cosTheta*colorC * Pss * (1.0f/fmax(outPdf, 1e-5f)), 0.0f);
    }
    else
    {
      const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hit.norm, hit.norm, 1.0f);
      const float  cosTheta = dot(newDir, hit.norm);
  
      const float pdfVal   = cosTheta * INV_PI;
      const float3 brdfVal = (cosTheta > 1e-5f) ? colorA * INV_PI : make_float3(0,0,0);
      const float3 bxdfVal = brdfVal * (1.0f / fmax(pdfVal, 1e-10f));
  
      rayPosAndNear[tid]    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
      rayDirAndFar[tid]     = to_float4(newDir, MAXFLOAT);
      accumThoroughput[tid] *= cosTheta*to_float4(bxdfVal, 0.0f);
    }
  }

#define PREFIX_SUMM_MACRO(idata,odata,l_Data,_bsize)       \
{                                                          \
  uint pos = 2 * get_local_id(0) - (get_local_id(0) & (_bsize - 1)); \
  l_Data[pos] = 0;                                         \
  pos += _bsize;                                           \
  l_Data[pos] = idata;                                     \
                                                           \
  for (uint offset = 1; offset < _bsize; offset <<= 1)     \
  {                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                          \
    uint t = l_Data[pos] + l_Data[pos - offset];           \
    barrier(CLK_LOCAL_MEM_FENCE);                          \
    l_Data[pos] = t;                                       \
  }                                                        \
                                                           \
  odata = l_Data[pos];                                     \
}                                                          \

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

static inline float3 SafeInverse_4to3(float4 d)
{
  const float ooeps = 1.0e-36f; // Avoid div by zero.
  float3 res;
  res.x = 1.0f / (fabs(d.x) > ooeps ? d.x : copysign(ooeps, d.x));
  res.y = 1.0f / (fabs(d.y) > ooeps ? d.y : copysign(ooeps, d.y));
  res.z = 1.0f / (fabs(d.z) > ooeps ? d.z : copysign(ooeps, d.z));
  return res;
}

static float2 RayBoxIntersectionLite(const float3 ray_pos, const float3 ray_dir_inv, const float boxMin[3], const float boxMax[3])
{
  const float lo = ray_dir_inv.x*(boxMin[0] - ray_pos.x);
  const float hi = ray_dir_inv.x*(boxMax[0] - ray_pos.x);

  float tmin = fmin(lo,hi);
  float tmax = fmax(lo,hi);

  const float lo1 = ray_dir_inv.y*(boxMin[1] - ray_pos.y);
  const float hi1 = ray_dir_inv.y*(boxMax[1] - ray_pos.y);

  tmin = fmax(tmin,fmin(lo1,hi1));
  tmax = fmin(tmax,fmax(lo1,hi1));

  const float lo2 = ray_dir_inv.z*(boxMin[2] - ray_pos.z);
  const float hi2 = ray_dir_inv.z*(boxMax[2] - ray_pos.z);

  tmin = fmax(tmin,fmin(lo2,hi2));
  tmax = fmin(tmax,fmax(lo2,hi2));

  return make_float2(tmin, tmax); //(tmin <= tmax) && (tmax > 0.f);
}

static void IntersectAllPrimitivesInLeaf(const float4 rayPosAndNear, const float4 rayDirAndFar,
                                         __global const uint* a_indices, uint a_start, uint a_count, __global const float4* a_vert,
                                         Lite_Hit* pHit, float2* pBars)
{
  const uint triAddressEnd = a_start + a_count;
  for (uint triAddress = a_start; triAddress < triAddressEnd; triAddress = triAddress + 3u)
  {
    const uint A = a_indices[triAddress + 0];
    const uint B = a_indices[triAddress + 1];
    const uint C = a_indices[triAddress + 2];

    const float4 A_pos = a_vert[A];
    const float4 B_pos = a_vert[B];
    const float4 C_pos = a_vert[C];

    const float4 edge1 = B_pos - A_pos;
    const float4 edge2 = C_pos - A_pos;
    const float4 pvec  = cross(rayDirAndFar, edge2);
    const float4 tvec  = rayPosAndNear - A_pos;
    const float4 qvec  = cross(tvec, edge1);
    const float dotTmp = dot(to_float3(edge1), to_float3(pvec));
    const float invDet = 1.0f / (dotTmp > 1e-6f ? dotTmp : 1e-6f);

    const float v = dot(to_float3(tvec), to_float3(pvec))*invDet;
    const float u = dot(to_float3(qvec), to_float3(rayDirAndFar))*invDet;
    const float t = dot(to_float3(edge2), to_float3(qvec))*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > rayPosAndNear.w && t < pHit->t)
    {
      pHit->t      = t;
      pHit->primId = triAddress/3;
      (*pBars)     = make_float2(u,v);
    }
  }

}

uint fakeOffset(uint x, uint y, uint pitch) { return y*pitch + x; }                                      // for 2D threading

#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16

/////////////////////////////////////////////////////////////////////
/////////////////// kernels /////////////////////////////////////////
/////////////////////////////////////////////////////////////////////


__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_GetColor(
  __global uint * out_color,
  __global uint* kgen_threadFlags,
  __global uint2       * kgen_objPtrData,
  __global unsigned int* kgen_objData,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= kgen_iNumElementsX)
    return;
  if((kgen_threadFlags[tid] & kgen_tFlagsMask) != 0) 
    return;
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint kgen_objPtr    = kgen_objPtrData[tid].x;
  const uint kgen_objTag    = (kgen_objPtr & IMaterial_TAG_MASK) >> (32 - IMaterial_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr & IMaterial_OFS_MASK);

  switch(kgen_objTag)
  {
    case IMaterial_TAG_LAMBERT: // implementation for LambertMaterial
    {
      __global LambertMaterial* pSelf = (__global LambertMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      LambertMaterial_kernel_GetColor(pSelf, tid, out_color, ubo);
    }
    break;
    case IMaterial_TAG_MIRROR: // implementation for PerfectMirrorMaterial
    {
      __global PerfectMirrorMaterial* pSelf = (__global PerfectMirrorMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      PerfectMirrorMaterial_kernel_GetColor(pSelf, tid, out_color, ubo);
    }
    break;
    case IMaterial_TAG_EMISSIVE: // implementation for EmissiveMaterial
    {
      __global EmissiveMaterial* pSelf = (__global EmissiveMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      EmissiveMaterial_kernel_GetColor(pSelf, tid, out_color, ubo);
    }
    break;
    case IMaterial_TAG_GGX_GLOSSY: // implementation for GGXGlossyMaterial
    {
      __global GGXGlossyMaterial* pSelf = (__global GGXGlossyMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      GGXGlossyMaterial_kernel_GetColor(pSelf, tid, out_color, ubo);
    }
    break;
    case IMaterial_TAG_LAMBERT_MIX: // implementation for LambertMaterialMix
    {
      __global LambertMaterialMix* pSelf = (__global LambertMaterialMix*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      LambertMaterialMix_kernel_GetColor(pSelf, tid, out_color, ubo);
    }
    break;
  default:
  break;
  };
}



__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_NextBounce(
  __global const Lite_Hit * in_hit,
  __global const float2 * in_bars,
  __global const uint32_t * in_indices,
  __global const float4 * in_vpos,
  __global const float4 * in_vnorm,
  __global float4 * rayPosAndNear,
  __global float4 * rayDirAndFar,
  __global RandomGen * pGen,
  __global float4 * accumColor,
  __global float4 * accumThoroughput,
  __global uint* kgen_threadFlags,
  __global uint2       * kgen_objPtrData,
  __global unsigned int* kgen_objData,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= kgen_iNumElementsX)
    return;
  if((kgen_threadFlags[tid] & kgen_tFlagsMask) != 0) 
    return;
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint kgen_objPtr    = kgen_objPtrData[tid].x;
  const uint kgen_objTag    = (kgen_objPtr & IMaterial_TAG_MASK) >> (32 - IMaterial_TAG_BITS);
  const uint kgen_objOffset = (kgen_objPtr & IMaterial_OFS_MASK);

  switch(kgen_objTag)
  {
    case IMaterial_TAG_LAMBERT: // implementation for LambertMaterial
    {
      __global LambertMaterial* pSelf = (__global LambertMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      LambertMaterial_kernel_NextBounce(pSelf, tid, in_hit, in_bars, in_indices, in_vpos, in_vnorm, rayPosAndNear, rayDirAndFar, pGen, accumColor, accumThoroughput);
    }
    break;
    case IMaterial_TAG_MIRROR: // implementation for PerfectMirrorMaterial
    {
      __global PerfectMirrorMaterial* pSelf = (__global PerfectMirrorMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      PerfectMirrorMaterial_kernel_NextBounce(pSelf, tid, in_hit, in_bars, in_indices, in_vpos, in_vnorm, rayPosAndNear, rayDirAndFar, pGen, accumColor, accumThoroughput);
    }
    break;
    case IMaterial_TAG_EMISSIVE: // implementation for EmissiveMaterial
    {
      __global EmissiveMaterial* pSelf = (__global EmissiveMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      EmissiveMaterial_kernel_NextBounce(pSelf, tid, in_hit, in_bars, in_indices, in_vpos, in_vnorm, rayPosAndNear, rayDirAndFar, pGen, accumColor, accumThoroughput);
    }
    break;
    case IMaterial_TAG_GGX_GLOSSY: // implementation for GGXGlossyMaterial
    {
      __global GGXGlossyMaterial* pSelf = (__global GGXGlossyMaterial*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      GGXGlossyMaterial_kernel_NextBounce(pSelf, tid, in_hit, in_bars, in_indices, in_vpos, in_vnorm, rayPosAndNear, rayDirAndFar, pGen, accumColor, accumThoroughput);
    }
    break;
    case IMaterial_TAG_LAMBERT_MIX: // implementation for LambertMaterialMix
    {
      __global LambertMaterialMix* pSelf = (__global LambertMaterialMix*)(kgen_objData + kgen_objOffset + 2); // '+ 2' due to vptr (assume 64 bit mode)
      LambertMaterialMix_kernel_NextBounce(pSelf, tid, in_hit, in_bars, in_indices, in_vpos, in_vnorm, rayPosAndNear, rayDirAndFar, pGen, accumColor, accumThoroughput);
    }
    break;
  default:
  break;
  };
}



__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_InitEyeRay(
  __global const uint * packedXY,
  __global float4 * rayPosAndNear,
  __global float4 * rayDirAndFar,
  __global uint* kgen_threadFlags,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= kgen_iNumElementsX)
    return;
  if((kgen_threadFlags[tid] & kgen_tFlagsMask) != 0) 
    return;
  const float3 camPos = ubo->camPos;
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  const float3 rayDir = EyeRayDir((float)x, (float)y, (float)WIN_WIDTH, (float)WIN_HEIGHT, ubo->m_worldViewProjInv); 
  const float3 rayPos = camPos;
  
  rayPosAndNear[tid] = to_float4(rayPos, 0.0f);
  rayDirAndFar[tid]  = to_float4(rayDir, MAXFLOAT);

}



__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_InitEyeRay2(
  __global const uint * packedXY,
  __global float4 * rayPosAndNear,
  __global float4 * rayDirAndFar,
  __global float4 * accumColor,
  __global float4 * accumuThoroughput,
  __global uint* kgen_threadFlags,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= kgen_iNumElementsX)
    return;
  if((kgen_threadFlags[tid] & kgen_tFlagsMask) != 0) 
    return;
  const float3 camPos = ubo->camPos;
  ///////////////////////////////////////////////////////////////// prolog
  
  accumColor[tid]        = make_float4(0,0,0,0);
  accumuThoroughput[tid] = make_float4(1,1,1,0);

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  const float3 rayDir = EyeRayDir((float)x, (float)y, (float)WIN_WIDTH, (float)WIN_HEIGHT, ubo->m_worldViewProjInv); 
  const float3 rayPos = camPos;
  
  rayPosAndNear[tid] = to_float4(rayPos, 0.0f);
  rayDirAndFar[tid]  = to_float4(rayDir, MAXFLOAT);

}



__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_RayTrace(
  __global const float4 * rayPosAndNear,
  __global float4 * rayDirAndFar,
  __global Lite_Hit * out_hit,
  __global float2 * out_bars,
  __global uint* kgen_threadFlags,
  __global float4* m_vPos4f,
  __global unsigned int* m_indicesReordered,
  __global struct Interval* m_intervals,
  __global struct BVHNode* m_nodes,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= kgen_iNumElementsX)
    return;
  if((kgen_threadFlags[tid] & kgen_tFlagsMask) != 0) 
    return;
  const float4 m_lightSphere = ubo->m_lightSphere;
  bool kgenExitCond = false;
  ///////////////////////////////////////////////////////////////// prolog
  
  const float4 rayPos = rayPosAndNear[tid];
  const float4 rayDir = rayDirAndFar[tid] ;

  const float3 rayDirInv = SafeInverse_4to3(rayDir);

  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;

  float2 baricentrics = make_float2(0,0);

  uint nodeIdx = 0;
  while(nodeIdx < 0xFFFFFFFE)
  {
    const struct BVHNode currNode = m_nodes[nodeIdx];
    const float2 boxHit           = RayBoxIntersectionLite(to_float3(rayPos), rayDirInv, currNode.boxMin, currNode.boxMax);
    const bool   intersects       = (boxHit.x <= boxHit.y) && (boxHit.y > rayPos.w) && (boxHit.x < res.t); // (tmin <= tmax) && (tmax > 0.f) && (tmin < curr_t)

    if(intersects && currNode.leftOffset == 0xFFFFFFFF) //leaf
    {
      struct Interval startCount = m_intervals[nodeIdx];
      IntersectAllPrimitivesInLeaf(rayPos, rayDir, m_indicesReordered, startCount.start*3, startCount.count*3, m_vPos4f, 
                                   &res, &baricentrics);
    }

    nodeIdx = (currNode.leftOffset == 0xFFFFFFFF || !intersects) ? currNode.escapeIndex : currNode.leftOffset;
    nodeIdx = (nodeIdx == 0) ? 0xFFFFFFFE : nodeIdx;
  }
  
  // intersect light under roof
  {
    const float2 tNearFar = RaySphereHit(to_float3(rayPos), to_float3(rayDir), m_lightSphere);
  
    if(tNearFar.x < tNearFar.y && tNearFar.x > 0.0f && tNearFar.x < res.t)
    {
      res.primId = 0;
      res.instId = -1;
      res.geomId = HIT_LIGHT_GEOM;
      res.t      = tNearFar.x;
    }
    else
      res.geomId = HIT_TRIANGLE_GEOM;
  }
  
  out_hit[tid]  = res;
  out_bars[tid] = baricentrics;
  kgenExitCond = (res.primId != -1); goto KGEN_EPILOG;

  KGEN_EPILOG:
  {
    const bool exitHappened = (kgen_tFlagsMask & KGEN_FLAG_SET_EXIT_NEGATIVE) != 0 ? !kgenExitCond : kgenExitCond;
    if((kgen_tFlagsMask & KGEN_FLAG_DONT_SET_EXIT) == 0 && exitHappened)
      kgen_threadFlags[tid] = ((kgen_tFlagsMask & KGEN_FLAG_BREAK) != 0) ? KGEN_FLAG_BREAK : KGEN_FLAG_RETURN;
  };
}


__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_MakeMaterial(
  __global const Lite_Hit * in_hit,
  __global uint* kgen_threadFlags,
  __global unsigned int* m_materialData,
  __global unsigned int* m_materialIds,
  __global unsigned int* m_materialOffsets,
  __global uint2       * kgen_objPtrData,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  uint kgen_objPtr = 0;
  const uint32_t m_emissiveMaterialId = ubo->m_emissiveMaterialId;
  ///////////////////////////////////////////////////////////////// prolog
  
  uint32_t objPtr = 0;  
 
  if(in_hit[tid].geomId == HIT_LIGHT_GEOM)
  {
    objPtr = m_materialOffsets[m_emissiveMaterialId];
  }
  else if(in_hit[tid].primId != -1)
  {
    const uint32_t mtId = m_materialIds[in_hit[tid].primId]+1; // +1 due to empty object
    objPtr = m_materialOffsets[mtId];
  }

  { kgen_objPtr = objPtr; };

  //KGEN_EPILOG:
  kgen_objPtrData[get_global_id(0)] = make_uint2(kgen_objPtr, get_global_id(0)); // put old threadId instead of zero
}


__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_PackXY(
  __global uint * out_pakedXY,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tidX = get_global_id(0); 
  const uint tidY = get_global_id(1); 
  if(tidX >= kgen_iNumElementsX || tidY >= kgen_iNumElementsY)
    return;
  ///////////////////////////////////////////////////////////////// prolog
  
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);

}



__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void kernel_ContributeToImage(
  __global const float4 * a_accumColor,
  __global const uint * in_pakedXY,
  __global float4 * out_color,
  __global uint* kgen_threadFlags,
  __global struct TestClass_UBO_Data* ubo,
  const uint kgen_iNumElementsX, 
  const uint kgen_iNumElementsY,
  const uint kgen_iNumElementsZ,
  const uint kgen_tFlagsMask)
{
  ///////////////////////////////////////////////////////////////// prolog
  const uint tid = get_global_id(0); 
  if(tid >= kgen_iNumElementsX)
    return;
  if((kgen_threadFlags[tid] & kgen_tFlagsMask) != 0) 
    return;
  ///////////////////////////////////////////////////////////////// prolog
  
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*WIN_WIDTH+x] += a_accumColor[tid];

}



__attribute__((reqd_work_group_size(256, 1, 1)))
__kernel void copyKernelFloat(
  __global float* out_data,
  __global float* in_data,
  const uint length)
{
  const uint i = get_global_id(0);
  if(i >= length)
    return;
  out_data[i] = in_data[i];
}

