#include "test_class.h"

#include <random>

static inline uint fakeOffset (uint x, uint y) { return 0; } 

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
   
  // generate spheres position and radius according to Tabula-Rasa
  //
  bool collision = false;
  int iterNum = 0;
  do
  {
    for(int i=0;i<a_numSpheres;i++)
    {
      spheresPosRadius[i] = make_float4(spheresPosDistr(generator),
                                        spheresPosDistr(generator),
                                        spheresPosDistr(generator)-10.0f,
                                        spheresRadDistr(generator));
    }
    
    collision = false;
    for(int i=0;i<a_numSpheres;i++)
    {
      const float4 sphereData1 = spheresPosRadius[i];
      const float3 spherePos1  = to_float3(sphereData1);
      const float  sphereR1    = sphereData1.w;
  
      for(int j = i+1;j<a_numSpheres;j++)
      {
        const float4 sphereData2 = spheresPosRadius[j];
        const float3 spherePos2  = to_float3(sphereData2);
        const float  sphereR2    = sphereData2.w;
  
        const float minDist = sphereR1+sphereR2;
        if(lengthSquare(spherePos1-spherePos2) < minDist*minDist)
        {
          collision = true;
          break;
        }
      }
  
      if(collision)
        break;
    }  

    iterNum++;
  } while (collision && iterNum < 1000);

  std::cout << "[InitSpheresScene]: Tabula-Rasa was applieds " << iterNum << " times" << std::endl;
}

void TestClass::kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const float3 rayDir = EyeRayDir((float)tidX, (float)tidY, (float)WIN_WIDTH, (float)WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = make_float3(0.0f, 0.0f, 0.0f);
  
  rayPosAndNear[fakeOffset(tidX,tidY)] = to_float4(rayPos, 0.0f);
  rayDirAndFar [fakeOffset(tidX,tidY)] = to_float4(rayDir, MAXFLOAT);
  flags        [fakeOffset(tidX,tidY)] = 0;
}

void TestClass::kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                                Lite_Hit* out_hit, uint tidX, uint tidY)
{
  const float3 rayPos = to_float3(rayPosAndNear[fakeOffset(tidX,tidY)]);
  const float3 rayDir = to_float3(rayDirAndFar[fakeOffset(tidX,tidY)]);

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

  out_hit     [fakeOffset(tidX,tidY)]   = res;
  rayDirAndFar[fakeOffset(tidX,tidY)].w = res.t;
}

void TestClass::kernel_TestColor(const Lite_Hit* in_hit, uint* out_color, uint tidX, uint tidY)
{
  const uint primId = in_hit[fakeOffset(tidX,tidY)].primId;

  if(primId != -1)
    out_color[pitchOffset(tidX,tidY)] = RealColorToUint32_f3(spheresMaterials[primId].color);
  else
    out_color[pitchOffset(tidX,tidY)] = 0x00000000;
}

void TestClass::CastSingleRay(uint tidX, uint tidY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;

  kernel_InitEyeRay(&flags, &rayPosAndNear, &rayDirAndFar, tidX, tidY);

  Lite_Hit hit;
  kernel_RayTrace(&rayPosAndNear, &rayDirAndFar, 
                  &hit, tidX, tidY);
  
  kernel_TestColor(&hit, 
                   out_color, tidX, tidY);
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
  
  for(int y=0;y<WIN_HEIGHT;y++)
  {
    for(int x=0;x<WIN_WIDTH;x++)
      test.CastSingleRay(x,y,pixelData.data());
  }

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  //std::cout << resultingColor.x << " " << resultingColor.y << " " << resultingColor.z << std::endl;
  return;
}