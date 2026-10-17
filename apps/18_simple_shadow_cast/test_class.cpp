#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>
#include <cfloat>

using std::max;
using std::min;

void TestClass::InitRandomGens(int a_maxThreads)
{
  m_randomGens.resize(a_maxThreads);
  #pragma omp parallel for default(shared)
  for(int i=0;i<a_maxThreads;i++)
    m_randomGens[i] = RandomGenInit(i);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Lite_Hit TestClass::RayTrace(float4 rayPos, float4 rayDir)
{
  CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
  
  Lite_Hit res;
  res.primId = hit.primId;
  res.instId = hit.instId;
  res.geomId = hit.geomId;
  res.t      = hit.t;

  // intersect flat light under roof
  {
    const float tLightHit  = (m_lightGeom.boxMax.y - rayPos.y)/max(rayDir.y, 1e-6f);
    const float4 hit_point = rayPos + tLightHit*rayDir;
    
    bool is_hit = (hit_point.x > m_lightGeom.boxMin.x) && (hit_point.x < m_lightGeom.boxMax.x) &&
                  (hit_point.z > m_lightGeom.boxMin.z) && (hit_point.z < m_lightGeom.boxMax.z) &&
                  (tLightHit < res.t);
  
    if(is_hit)
    {
      res.primId = 0;
      res.instId = -1;
      res.geomId = HIT_FLAT_LIGHT_GEOM;
      res.t      = tLightHit;
    }
    else
      res.geomId = HIT_TRIANGLE_GEOM;
  }

  return res;
}

float4 TestClass::ShadowRayOrigin(float4 rayPos, float4 rayDir, float hitT)
{
  // dirty hack to offset shadow ray origin towards the light
  //
  const float3 hitPos    = to_float3(rayPos) + hitT*0.99995f*to_float3(rayDir);
  const float3 boxCenter = 0.5f*(to_float3(m_lightGeom.boxMin) + to_float3(m_lightGeom.boxMax));
  return to_float4(hitPos + 1e-6f*normalize(boxCenter - hitPos), 0.0f);
}

float TestClass::CalcShadow(uint tid, float4 surfPos)
{
  RandomGen gen = m_randomGens[tid];

  float shadow = 0.0f;
  const int numSamples = 16;
  
  for(int i=0;i<numSamples;i++)
  {
    const float2 uv = rndFloat2_Pseudo(&gen);
    const float3 samplePos  = to_float3(m_lightGeom.boxMin) + float3(uv.x, 0.5f, uv.y)*to_float3(m_lightGeom.boxMax - m_lightGeom.boxMin);
    
    const float3 shadowRayPos = to_float3(surfPos);
    const float  hitDist      = length(samplePos - shadowRayPos);
    const float3 shadowRayDir = normalize(samplePos - shadowRayPos);

    const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  
    if(inShadow)
      shadow += 1.0f;
  }

  return 1.0f - shadow*(1.0f/float(numSamples));  
}

uint TestClass::GetTestColor(Lite_Hit hit, float shadow)
{
  if(hit.geomId == HIT_FLAT_LIGHT_GEOM)
    return RealColorToUint32_f3(float3(1,1,1));
  
  const uint32_t mtId = m_materialIds[hit.primId];
  const float4 mdata  = m_materials[mtId];
  return RealColorToUint32_f3(to_float3(mdata)*(0.1f + 0.9f*shadow)); 
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::kernel2D_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  #pragma omp parallel for default(shared)
  for(uint y=0;y<tidY;y++)
  {
    for(uint x=0;x<tidX;x++)
      out_pakedXY[pitchOffset(x,y)] = ((y << 16) & 0xFFFF0000) | (x & 0x0000FFFF);
  }
}

void TestClass::kernel1D_CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color)
{
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
  {
    const uint XY = in_pakedXY[i];
    const uint x  = (XY & 0x0000FFFF);
    const uint y  = (XY & 0xFFFF0000) >> 16;

    const float4 rayPos = to_float4(m_camPos, 0.0f);
    const float4 rayDir = to_float4(EyeRayDir(x, y, WIN_WIDTH, WIN_HEIGHT, m_worldViewProjInv), FLT_MAX);
    
    const Lite_Hit hit = RayTrace(rayPos, rayDir);
    if((hit.primId != -1) && (hit.t < rayDir.w))
    {
      const float4 surfPos = ShadowRayOrigin(rayPos, rayDir, hit.t);
      const float  shadow  = CalcShadow(i, surfPos);
      out_color[i] = GetTestColor(hit, shadow);
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel2D_PackXY(tidX, tidY, out_pakedXY);
}

void TestClass::CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color)
{
  auto start = std::chrono::high_resolution_clock::now();
  kernel1D_CastSingleRay(tid, in_pakedXY, out_color);
  m_executionTimeCast = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "CastSingleRay")
    a_out[0] = m_executionTimeCast;
}
