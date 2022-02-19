#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

float Integrator::LightPdfSelectRev(int a_lightId) 
{ 
  return 1.0f; 
}

float Integrator::LightEvalPDF(int a_lightId, float3 illuminationPoint, float3 ray_dir, const SurfaceHit* pSurfaceHit)
{
  const float3 lpos   = pSurfaceHit->pos;
  const float3 lnorm  = pSurfaceHit->norm;
  const float hitDist = length(illuminationPoint - lpos);
  const float pdfA    = 1.0f / (4.0f*m_light.size.x*m_light.size.y);
  const float cosVal  = std::max(dot(ray_dir, -1.0f*lnorm), 0.0f);
  return PdfAtoW(pdfA, hitDist, cosVal);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float3 Integrator::MaterialSample(int a_materialId, float2 rands, float3 v, float3 n)
{
  uint type = m_materials[a_materialId].brdfType;
  if(type == BRDF_TYPE_GGX)
    return ggxSample(rands, v, n, 1.0f - m_materials[a_materialId].glosiness);
  else
    return lambertSample(rands, v, n);
}

float Integrator::MaterialEvalPDF(int a_materialId, float3 l, float3 v, float3 n) 
{ 
  uint type = m_materials[a_materialId].brdfType;
  if(type == BRDF_TYPE_GGX)
    return ggxEvalPDF(l, v, n, 1.0f - m_materials[a_materialId].glosiness);
  else
    return lambertEvalPDF(l, v, n);
}

float3 Integrator::MaterialEvalBSDF(int a_materialId, float3 l, float3 v, float3 n)
{
  if(std::abs(dot(l, n)) < 1e-5f)
    return float3(0,0,0); 

  uint type = m_materials[a_materialId].brdfType;
  if(type == BRDF_TYPE_GGX)
  {
    const float3 color = float3(m_materials[a_materialId].reflection[0], 
                                m_materials[a_materialId].reflection[1], 
                                m_materials[a_materialId].reflection[2]); 
  
    return color*ggxEvalBSDF(l, v, n, 1.0f - m_materials[a_materialId].glosiness);
  }
  else
  {
    const float3 color = float3(m_materials[a_materialId].diffuse[0], 
                                m_materials[a_materialId].diffuse[1], 
                                m_materials[a_materialId].diffuse[2]); 
  
    return color*lambertEvalBSDF(l, v, n);
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel_PackXY(tidX, tidY, out_pakedXY);
}

void Integrator::CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit; 
  float2   baricentrics; 
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return;
  
  kernel_GetRayColor(tid, &hit, in_pakedXY, out_color);
}

void Integrator::NaivePathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &rayFlags);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_ContributeToImage(tid, &accumColor, &gen, in_pakedXY, 
                           out_color);
}

void Integrator::PathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &rayFlags);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags, 
                             &gen, &shadeColor);

    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_ContributeToImage(tid, &accumColor, &gen, in_pakedXY, 
                           out_color);
                           
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum)
{
  #pragma omp parallel for default(shared)
  for(int y=0;y<tidY;y++)
    for(int x=0;x<tidX;x++)
      PackXY(x, y, out_pakedXY);
}

void Integrator::CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum)
{
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    CastSingleRay(i, in_pakedXY, out_color);
}

void Integrator::NaivePathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passNum;j++)
      NaivePathTrace(i, 6, in_pakedXY, out_color);
  naivePtTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void Integrator::PathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passNum;j++)
      PathTrace(i, 6, in_pakedXY, out_color);
  shadowPtTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void Integrator::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "NaivePathTrace" || std::string(a_funcName) == "NaivePathTraceBlock")
    a_out[0] = naivePtTime;
  else if(std::string(a_funcName) == "PathTrace" || std::string(a_funcName) == "PathTraceBlock")
    a_out[0] = shadowPtTime;
}
