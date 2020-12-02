#include "test_class.h"
#include "include/crandom.h"

#include <random>

static inline uint fakeOffset (uint x, uint y) { return 0; } 
static inline uint fakeOffset (uint x) { return 0; } 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::InitSpheresScene(int a_numSpheres, int a_seed)
{
  srand(a_seed);

  spheresPosRadius.resize(a_numSpheres);
  spheresMaterials.resize(a_numSpheres);

  std::default_random_engine generator(a_seed);
  std::uniform_real_distribution<float> distrColors(0.25f, 0.75f);
  std::uniform_real_distribution<float> spheresPosDistr(-5.0f, 5.0f);
  std::uniform_real_distribution<float> spheresRadDistr(0.25f, 3.0f);
  
  // generate material properties first
  //
  for(int i=0;i<a_numSpheres;i++)
  {
    bool isEmissive = (i % 5) == 0;
    if(isEmissive)
    {
      spheresMaterials[i].flags = MTL_EMISSIVE;
      spheresMaterials[i].color = float3(10,10,10);
    }
    else
    {
      spheresMaterials[i].flags = 0;
      spheresMaterials[i].color = float3(0.25f + distrColors(generator), 
                                         0.25f + distrColors(generator), 
                                         0.25f + distrColors(generator));
    }
  }
   
  for(int i=0;i<a_numSpheres;i++)
  {
    spheresPosRadius[i] = make_float4(spheresPosDistr(generator),
                                      spheresPosDistr(generator),
                                      spheresPosDistr(generator)-10.0f,
                                      spheresRadDistr(generator));
  }
  
}

void TestClass::InitRandomGens(int a_maxThreads)
{
  m_randomGens.resize(a_maxThreads);
  for(int i=0;i<a_maxThreads;i++)
    m_randomGens[i] = RandomGenInit(i);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::kernel_InitEyeRay(uint tid, const uint* packedXY, uint* flags, float4* rayPosAndNear, float4* rayDirAndFar) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  const float3 rayDir = EyeRayDir((float)x, (float)y, (float)WIN_WIDTH, (float)WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = make_float3(0.0f, 0.0f, 0.0f);
  
  rayPosAndNear[fakeOffset(tid)] = to_float4(rayPos, 0.0f);
  rayDirAndFar [fakeOffset(tid)] = to_float4(rayDir, MAXFLOAT);
  flags        [fakeOffset(tid)] = 0;
}

void TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar, const uint* flags,
                                Lite_Hit* out_hit)
{
  if(flags[fakeOffset(tid)] & THREAD_IS_DEAD != 0)
    return;

  const float3 rayPos = to_float3(rayPosAndNear[fakeOffset(tid)]);
  const float3 rayDir = to_float3(rayDirAndFar [fakeOffset(tid)]);

  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = MAXFLOAT;
  
  for(int sphereId=0;sphereId<spheresPosRadius.size();sphereId++)
  {
    const float2 tNearAndFar = RaySphereHit(rayPos, rayDir, spheresPosRadius[sphereId]);
  
    if(tNearAndFar.x < tNearAndFar.y && tNearAndFar.x < res.t)
    {
      res.t      = tNearAndFar.x;
      res.primId = sphereId;
    }
  }

  out_hit     [fakeOffset(tid)]   = res;
  rayDirAndFar[fakeOffset(tid)].w = res.t;
}

void TestClass::kernel_GetMaterialColor(uint tid, const Lite_Hit* in_hit, 
                                        uint* out_color)
{
  const uint primId = in_hit[fakeOffset(tid)].primId;

  if(primId != -1)
    out_color[tid] = RealColorToUint32_f3(spheresMaterials[primId].color);
  else
    out_color[tid] = 0x00000000;
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_InitAccumData(uint tid, float4* accumColor, float4* accumuThoroughput)
{
  accumColor       [fakeOffset(tid)] = make_float4(0,0,0,0);
  accumuThoroughput[fakeOffset(tid)] = make_float4(1,1,1,0);
}

void TestClass::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(a_accumColor[fakeOffset(tid)]);
}

void TestClass::kernel_NextBounce(uint tid, const Lite_Hit* in_hit, 
                                  uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput)
{
  if(flags[fakeOffset(tid)] & THREAD_IS_DEAD != 0)
    return;

  const Lite_Hit hit = in_hit[fakeOffset(tid)];
  if(hit.primId == -1)
  {
    //accumColor[fakeOffset(tid)] = m_envColor*accumThoroughput[fakeOffset(tid)];
    flags[fakeOffset(tid)] = THREAD_IS_DEAD;
    return;
  }

  const float3 rayPos = to_float3(rayPosAndNear[fakeOffset(tid)]);
  const float3 rayDir = to_float3(rayDirAndFar[fakeOffset(tid)]);

  if( IsMtlEmissive(&spheresMaterials[hit.primId]) )
  {
    float4 emissiveColor = to_float4(GetMtlEmissiveColor(&spheresMaterials[hit.primId]), 0.0f);
    accumColor[fakeOffset(tid)] = emissiveColor*accumThoroughput[fakeOffset(tid)];
    flags     [fakeOffset(tid)] = THREAD_IS_DEAD;
    return;
  }

  const float3 sphPos    = to_float3(spheresPosRadius[hit.primId]);
  const float3 diffColor = GetMtlDiffuseColor(&spheresMaterials[hit.primId]);

  const float3 hitPos  = rayPos + rayDir*hit.t;
  const float3 hitNorm = normalize(hitPos - sphPos);

  RandomGen gen = m_randomGens[fakeOffset(tid)];
  const float2 uv = rndFloat2_Pseudo(&gen);
  m_randomGens[fakeOffset(tid)] = gen;

  const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hitNorm, hitNorm, 1.0f);
  const float  cosTheta = dot(newDir, hitNorm);

  const float pdfVal   = cosTheta * INV_PI;
  const float3 brdfVal = (cosTheta > 1e-5f) ? diffColor * INV_PI : make_float3(0,0,0);
  const float3 bxdfVal = brdfVal * (1.0f / fmax(pdfVal, 1e-10f));

  const float3 newPos = OffsRayPos(hitPos, hitNorm, newDir);  

  rayPosAndNear   [fakeOffset(tid)] = to_float4(newPos, 0.0f);
  rayDirAndFar    [fakeOffset(tid)] = to_float4(newDir, MAXFLOAT);
  accumThoroughput[fakeOffset(tid)] *= cosTheta*to_float4(bxdfVal, 0.0f);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*WIN_WIDTH+x] += a_accumColor[fakeOffset(tid)];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel_PackXY(tidX, tidY, out_pakedXY);
}

void TestClass::CastSingleRay(uint tid, uint* in_pakedXY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;
  kernel_InitEyeRay(tid, in_pakedXY, &flags, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit;
  kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &flags,
                  &hit);
  
  kernel_GetMaterialColor(tid, &hit, out_color);
}

void TestClass::StupidPathTrace(uint tid, uint a_maxDepth, uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  kernel_InitAccumData(tid, &accumColor, &accumThoroughput);

  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;
  kernel_InitEyeRay(tid, in_pakedXY, &flags, &rayPosAndNear, &rayDirAndFar);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &flags,
                    &hit);

    kernel_NextBounce(tid, &hit, 
                      &flags, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput);
  }

  kernel_ContributeToImage(tid, &accumColor, in_pakedXY, 
                           out_color);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Bitmap.h"

void test_class_cpu()
{
  TestClass test;

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
  
  // test simple ray casting
  //
  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    test.CastSingleRay(i, packedXY.data(), pixelData.data());

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  // now test path tracing
  //
  const int PASS_NUMBER = 100;
  for(int passId = 0; passId < PASS_NUMBER; passId++)
  {
    for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
      test.StupidPathTrace(i, 5, packedXY.data(), realColor.data());

    if(passId%10 == 0)
    {
      const float progress = 100.0f*float(passId)/float(PASS_NUMBER);
      std::cout << "progress = " << progress << "%   \r";
      std::cout.flush();
    }
  }
  
  std::cout << std::endl;

  const float normConst = 1.0f/float(PASS_NUMBER);

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    pixelData[i] = RealColorToUint32(realColor[i]*normConst);

  SaveBMP("zout_cpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  //std::cout << resultingColor.x << " " << resultingColor.y << " " << resultingColor.z << std::endl;
  return;
}