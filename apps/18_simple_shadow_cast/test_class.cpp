#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

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

void TestClass::kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  const float3 rayDir = EyeRayDir(x, y, WIN_WIDTH, WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = m_camPos;
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}

bool TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                Lite_Hit* out_hit, float4* out_surfPos)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
  
  Lite_Hit res;
  res.primId = hit.primId;
  res.instId = hit.instId;
  res.geomId = hit.geomId;
  res.t      = hit.t;

  float2 baricentrics = float2(hit.coords[0], hit.coords[1]);

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
 
  *out_hit  = res;
  
  // dirty hack to offset shadow ray in next kernel
  //
  const float3 hitPos    = to_float3(rayPos) + res.t*0.99995f*to_float3(rayDir);
  const float3 boxCenter = 0.5f*(m_lightGeom.boxMin + m_lightGeom.boxMax);
  (*out_surfPos)         = to_float4(hitPos + 1e-6f*normalize(boxCenter - hitPos), 0.0f);

  return (res.primId != -1) && (res.t < rayDir.w);
}

void TestClass::kernel_CalcShadow(uint tid, const float4* in_pos, float* out_shadow)
{
  RandomGen gen = m_randomGens[tid];

  float shadow = 0.0f;
  const int numSamples = 16;
  
  for(int i=0;i<numSamples;i++)
  {
    const float2 uv = rndFloat2_Pseudo(&gen);
    const float3 samplePos  = m_lightGeom.boxMin + float3(uv.x, 0.5f, uv.y)*(m_lightGeom.boxMax - m_lightGeom.boxMin);
    
    const float3 shadowRayPos = to_float3(*in_pos);
    const float  hitDist      = length(samplePos - shadowRayPos);
    const float3 shadowRayDir = normalize(samplePos - shadowRayPos);

    const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  
    if(inShadow)
      shadow += 1.0f;
  }

  *out_shadow = 1.0f - shadow*(1.0f/float(numSamples));  
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_GetTestColor(uint tid, const Lite_Hit* in_hit, const float* in_shadow, uint* out_color)
{
  if(in_hit->geomId == HIT_FLAT_LIGHT_GEOM)
  {
    out_color[tid] = RealColorToUint32_f3(float3(1,1,1));
  }
  else
  {
    const uint32_t mtId = m_materialIds[in_hit->primId];
    const float4 mdata  = m_materials[mtId];
    const float shadow  = *in_shadow;
    out_color[tid]      = RealColorToUint32_f3(to_float3(mdata)*(0.1f + 0.9f*shadow)); 
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel_PackXY(tidX, tidY, out_pakedXY);
}

void TestClass::CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit; 
  float4   surfPos; 
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &surfPos))
    return;
  
  float shadow;
  kernel_CalcShadow(tid, &surfPos, &shadow);

  kernel_GetTestColor(tid, &hit, &shadow, out_color);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum)
{
  #pragma omp parallel for default(shared)
  for(int y=0;y<tidY;y++)
    for(int x=0;x<tidX;x++)
      PackXY(x, y, out_pakedXY);
}

void TestClass::CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    CastSingleRay(i, in_pakedXY, out_color);
  m_executionTimeCast = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "CastSingleRay" || std::string(a_funcName) == "CastSingleRayBlock")
    a_out[0] = m_executionTimeCast;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Bitmap.h"

void test_class_cpu()
{
  TestClass test(WIN_WIDTH*WIN_HEIGHT);

  std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
  std::vector<uint32_t> packedXY(WIN_WIDTH*WIN_HEIGHT);
  std::vector<float4>   realColor(WIN_WIDTH*WIN_HEIGHT);
  
  // remember pitch-linear (x,y) for each thread to make our threading 1D
  //
  for(int y=0;y<WIN_HEIGHT;y++)
  {
    for(int x=0;x<WIN_WIDTH;x++)
      test.PackXY(x, y, packedXY.data());
  }

  test.LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.vsgf");

  // test simple ray casting
  //
  #pragma omp parallel for default(shared)
  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    test.CastSingleRay(i, packedXY.data(), pixelData.data());

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
}
