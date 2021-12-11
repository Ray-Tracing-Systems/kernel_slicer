#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::InitSpheresScene(int a_numSpheres, int a_seed)
{ 
  spheresPosRadius.resize(8);
  spheresMaterials.resize(8);

  spheresPosRadius[0] = float4(0,-10000.0f,0,9999.0f);
  spheresMaterials[0].color = float4(0.5,0.5,0.5, 0.0f);

  spheresPosRadius[1] = float4(0,0,-4,1);
  spheresMaterials[1].color = float4(1,1,1,5);

  const float col = 0.75f;
  const float eps = 0.00f;

  spheresPosRadius[2] = float4(-2,0,-4,1);
  spheresMaterials[2].color = float4(col,eps,eps,0);

  spheresPosRadius[3] = float4(+2,0,-4,1);
  spheresMaterials[3].color = float4(eps,col,col,0);

  spheresPosRadius[4] = float4(-1,1.5,-4.5,1);
  spheresMaterials[4].color = float4(col,col,eps,0);

  spheresPosRadius[5] = float4(+1,1.5,-4.5,1);
  spheresMaterials[5].color = float4(eps,eps,col,0);

  spheresPosRadius[6] = float4(-1,-0.5,-3,0.5);
  spheresMaterials[6].color = float4(eps,col,eps,0);

  spheresPosRadius[7] = float4(+1,-0.5,-3,0.5);
  spheresMaterials[7].color = float4(eps,col,eps,0);
}

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

  const float3 rayDir = EyeRayDir((float)x, (float)y, (float)WIN_WIDTH, (float)WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = make_float3(0.0f, 0.0f, 0.0f);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}

bool TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                Lite_Hit* out_hit)
{
  const float3 rayPos = to_float3(*rayPosAndNear);
  const float3 rayDir = to_float3(*rayDirAndFar );

  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = MAXFLOAT;
  
  for(int sphereId=0;sphereId<spheresPosRadius.size();sphereId++)
  {
    const float2 tNearAndFar = RaySphereHit(rayPos, rayDir, spheresPosRadius[sphereId]);
  
    if(tNearAndFar.x < tNearAndFar.y && tNearAndFar.x > 0.0f && tNearAndFar.x < res.t)
    {
      res.t      = tNearAndFar.x;
      res.primId = sphereId;
    }
  }

  *out_hit = res;
  return (res.primId != -1);
}

void TestClass::kernel_GetMaterialColor(uint tid, const Lite_Hit* in_hit, 
                                        uint* out_color)
{
  if(in_hit->primId != -1)
    out_color[tid] = RealColorToUint32_f3(to_float3(spheresMaterials[in_hit->primId].color));
  else
    out_color[tid] = 0x00700000;
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_InitAccumData(uint tid, float4* accumColor, float4* accumuThoroughput)
{
  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,0);
}

void TestClass::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(*a_accumColor);
}

void TestClass::kernel_NextBounce(uint tid, const Lite_Hit* in_hit, 
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput)
{
  const Lite_Hit hit  = *in_hit;
  const float3 rayPos = to_float3(*rayPosAndNear);
  const float3 rayDir = to_float3(*rayDirAndFar );

  if( IsMtlEmissive(&spheresMaterials[hit.primId]) )
  {
    float4 emissiveColor = to_float4(GetMtlEmissiveColor(&spheresMaterials[hit.primId]), 0.0f);
    *accumColor = emissiveColor*(*accumThoroughput);
    return;
  }

  const float3 sphPos    = to_float3(spheresPosRadius[hit.primId]);
  const float3 diffColor = GetMtlDiffuseColor(&spheresMaterials[hit.primId]);

  const float3 hitPos  = rayPos + rayDir*hit.t;
  const float3 hitNorm = normalize(hitPos - sphPos);

  RandomGen gen = m_randomGens[tid];
  const float2 uv = rndFloat2_Pseudo(&gen);
  m_randomGens[tid] = gen;

  const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hitNorm, hitNorm, 1.0f);
  const float  cosTheta = dot(newDir, hitNorm);

  const float pdfVal   = cosTheta * INV_PI;
  const float3 brdfVal = (cosTheta > 1e-5f) ? diffColor * INV_PI : make_float3(0,0,0);
  const float3 bxdfVal = brdfVal * (1.0f / fmax(pdfVal, 1e-10f));

  const float3 newPos = OffsRayPos(hitPos, hitNorm, newDir);  

  *rayPosAndNear    = to_float4(newPos, 0.0f);
  *rayDirAndFar     = to_float4(newDir, MAXFLOAT);
  *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*WIN_WIDTH+x] += *a_accumColor;
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
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar,
                      &hit))
    return;
  
  kernel_GetMaterialColor(tid, &hit, out_color);
}

void TestClass::StupidPathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  kernel_InitAccumData(tid, &accumColor, &accumThoroughput);

  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit))
      break;

    kernel_NextBounce(tid, &hit, 
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput);
  }

  kernel_ContributeToImage(tid, &accumColor, in_pakedXY, 
                           out_color);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// please note that further code is not processed by kslicer, it is just CPU code \\

void TestClass::PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passesNum)
{
  #pragma omp parallel for default(shared)
  for(int y=0;y<tidY;y++)
    for(int x=0;x<tidX;x++)
      PackXY(x, y, out_pakedXY);
}

void TestClass::CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passesNum)
{
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    CastSingleRay(i, in_pakedXY, out_color);
}

void TestClass::StupidPathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passesNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passesNum;j++)
      StupidPathTrace(i, 6, in_pakedXY, out_color);
  m_executionTimePT = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "StupidPathTrace" || std::string(a_funcName) == "StupidPathTraceBlock")
    a_out[0] = m_executionTimePT;
}