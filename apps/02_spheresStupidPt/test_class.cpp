#include "test_class.h"

#include <random>

static inline uint fakeOffset (uint x, uint y) { return 0; } 
static inline uint fakeOffset (uint x) { return 0; } 

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

void TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar, 
                                Lite_Hit* out_hit)
{
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

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel_PackXY(tidX, tidY, out_pakedXY);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::CastSingleRay(uint tid, uint* in_pakedXY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;

  kernel_InitEyeRay(tid, in_pakedXY, &flags, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit;
  kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, 
                  &hit);
  
  kernel_GetMaterialColor(tid, &hit, out_color);
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
  
  for(int y=0;y<WIN_HEIGHT;y++)
  {
    for(int x=0;x<WIN_WIDTH;x++)
      test.PackXY(x, y, packedXY.data());
  }

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    test.CastSingleRay(i, packedXY.data(),
                       pixelData.data());
  }

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  //std::cout << resultingColor.x << " " << resultingColor.y << " " << resultingColor.z << std::endl;
  return;
}