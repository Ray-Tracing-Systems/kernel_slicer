#include <vector>
#include <chrono>
#include <cfloat>

#include "test_class.h"

TestClass::TestClass(int w, int h) 
{
  m_pRayTraceImpl = std::make_shared<BFRayTrace>(); 
  m_pRayTraceImpl->InitBoxesAndTris(60,25); 

  m_widthInv  = 1.0f/float(w); 
  m_heightInv = 1.0f/float(h); 
}

static inline float2 RayBoxIntersection2(float3 rayOrigin, float3 rayDirInv, float3 boxMin, float3 boxMax)
{
  const float lo  = rayDirInv.x*(boxMin.x - rayOrigin.x);
  const float hi  = rayDirInv.x*(boxMax.x - rayOrigin.x);
  const float lo1 = rayDirInv.y*(boxMin.y - rayOrigin.y);
  const float hi1 = rayDirInv.y*(boxMax.y - rayOrigin.y);
  const float lo2 = rayDirInv.z*(boxMin.z - rayOrigin.z);
  const float hi2 = rayDirInv.z*(boxMax.z - rayOrigin.z);

  const float tmin = std::max(std::min(lo, hi), std::min(lo1, hi1));
  const float tmax = std::min(std::max(lo, hi), std::max(lo1, hi1));

  return float2(std::max(tmin, std::min(lo2, hi2)), 
                std::min(tmax, std::max(lo2, hi2)));
}

void TestClass::kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY) 
{
  const float x = float(tidX)*m_widthInv;
  const float y = float(tidY)*m_heightInv;
  *(rayPosAndNear) = float4(x, y, -1.0f, 0.0f);
  *(rayDirAndFar ) = float4(0, 0, 1, FLT_MAX);
  *flags           = 0;
}

void TestClass::kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                                int* out_hit, uint tidX, uint tidY)
{
  int hitId = m_pRayTraceImpl->RayTrace(*rayPosAndNear, *rayDirAndFar); 
  *out_hit  = hitId;
}

void TestClass::kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY)
{
  if(*in_hit != -1)
    out_color[pitchOffset(tidX,tidY)] = 0x000000FF;
  else
    out_color[pitchOffset(tidX,tidY)] = 0x00FF0000;
}


void TestClass::BFRT_ReadAndCompute(uint tidX, uint tidY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;
  int hit;

  kernel_InitEyeRay(&flags, &rayPosAndNear, &rayDirAndFar, tidX, tidY);
  kernel_RayTrace  (&rayPosAndNear, &rayDirAndFar, &hit, tidX, tidY);
  kernel_TestColor (&hit, out_color, tidX, tidY);
}

void TestClass::BFRT_ReadAndComputeBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses)
{
  auto before = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for collapse(2)
  for(int y=0;y<tidY;y++)
    for(int x=0;x<tidX;x++)
      for(int p=0;p<a_numPasses;p++)
        BFRT_ReadAndCompute(x,y,out_color);
  m_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}


void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName).find("ReadAndCompute") != std::string::npos)
    a_out[0] = m_time1;
  else if(std::string(a_funcName).find("Compute") != std::string::npos)
    a_out[0] = m_time2;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void BFRayTrace::InitBoxesAndTris(int numBoxes, int numTris)
{
  boxes.resize(numBoxes*2);
  trivets.resize(numTris*3);

  int boxesInString = std::max(numBoxes/5, 5);
  int trisInString  = std::max(numTris/5, 5);

  float boxSize = 0.25f/float(boxesInString);
  float triSize = 0.25f/float(trisInString);

  // (1) first half of the screen contain boxes
  //
  for(int i=0;i<numBoxes;i++)
  { 
    int centerX = i%boxesInString;
    int centerY = i/boxesInString;
    boxes[i*2+0] = float4(float(centerX + 0.5f)/float(boxesInString) - boxSize, float(centerY + 0.5f)/float(boxesInString) - boxSize, 1.0f*i + 0.0f, 0.0f);
    boxes[i*2+1] = float4(float(centerX + 0.5f)/float(boxesInString) + boxSize, float(centerY + 0.5f)/float(boxesInString) + boxSize, 1.0f*i + 0.5f, 0.0f);
  }

  // (2) second half of the screen contain triangles
  //
  for(int i=0;i<numTris;i++)
  {
    int centerX = i%trisInString;
    int centerY = (numBoxes/boxesInString) + (i)/trisInString;

    trivets[i*3+0] = float4(float(centerX + 0.75f)/float(boxesInString) - boxSize, float(centerY + 0.75f)/float(boxesInString) - boxSize, i, 0.0f);
    trivets[i*3+1] = float4(float(centerX + 0.25f)/float(boxesInString),           float(centerY + 0.75f)/float(boxesInString), i, 0.0f);
    trivets[i*3+2] = float4(float(centerX + 0.75f)/float(boxesInString) - boxSize, float(centerY + 0.75f)/float(boxesInString) + boxSize, i, 0.0f);
  }
}

int  BFRayTrace::RayTrace(float4 rayPosAndNear, float4 rayDirAndFar)
{
  const float3 rayPos    = to_float3(rayPosAndNear);
  const float3 rayDirInv = 1.0f/to_float3(rayDirAndFar);
  
  const float tNear = rayPosAndNear.w;
  const float tFar  = rayDirAndFar.w + testOffset;
  
  int hitId = -1;
  for(uint32_t boxId = 0; boxId < boxes.size(); boxId+=4) 
  {
    const float2 tm0 = RayBoxIntersection2(rayPos, rayDirInv, to_float3(boxes[boxId+0]), to_float3(boxes[boxId+1]));
    const float2 tm1 = RayBoxIntersection2(rayPos, rayDirInv, to_float3(boxes[boxId+2]), to_float3(boxes[boxId+3]));

    const bool hitChild0 = (tm0.x <= tm0.y) && (tm0.y >= tNear) && (tm0.x <= tFar);
    const bool hitChild1 = (tm1.x <= tm1.y) && (tm1.y >= tNear) && (tm1.x <= tFar);

    if(hitChild0)
      hitId = int(boxId >> 2);
    else if(hitChild1)
      hitId = int((boxId+2) >> 2);
  }

  const float3 ray_dir = to_float3(rayDirAndFar);
  for (uint32_t triId = 0; triId < trivets.size(); triId+=3)
  {
    const float3 A_pos = to_float3(trivets[triId + 0]);
    const float3 B_pos = to_float3(trivets[triId + 1]);
    const float3 C_pos = to_float3(trivets[triId + 2]);

    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec = cross(ray_dir, edge2);
    const float3 tvec = rayPos - A_pos;
    const float3 qvec = cross(tvec, edge1);

    const float invDet = 1.0f / dot(edge1, pvec);
    const float v = dot(tvec, pvec) * invDet;
    const float u = dot(qvec, ray_dir) * invDet;
    const float t = dot(edge2, qvec) * invDet;

    if (v >= -1e-6f && u >= -1e-6f && (u + v <= 1.0f + 1e-6f) && t > tNear && t < tFar)
      hitId = int(triId);
  }

  return hitId;
}
