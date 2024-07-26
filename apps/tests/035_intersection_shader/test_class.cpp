#include <vector>
#include <chrono>
#include <cfloat>

#include "test_class.h"

TestClass::TestClass(int w, int h) 
{
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

static inline float2 RaySphereHit(float3 orig, float3 dir, float4 sphere) // see Ray Tracing Gems Book
{
  const float3 center = to_float3(sphere);
  const float  radius = sphere.w;

  // Hearn and Baker equation 10-72 for when radius^2 << distance between origin and center
	// Also at https://www.cg.tuwien.ac.at/courses/EinfVisComp/Slides/SS16/EVC-11%20Ray-Tracing%20Slides.pdf
	// Assumes ray direction is normalized
	//dir = normalize(dir);
	const float3 deltap   = center - orig;
	const float ddp       = dot(dir, deltap);
	const float deltapdot = dot(deltap, deltap);

	// old way, "standard", though it seems to be worse than the methods above
	//float discriminant = ddp * ddp - deltapdot + radius * radius;
	float3 remedyTerm  = deltap - ddp * dir;
	float discriminant = radius * radius - dot(remedyTerm, remedyTerm);

  float2 result = {0,0};
	if (discriminant >= 0.0f)
	{
		const float sqrtVal = std::sqrt(discriminant);
		// include Press, William H., Saul A. Teukolsky, William T. Vetterling, and Brian P. Flannery, 
		// "Numerical Recipes in C," Cambridge University Press, 1992.
		const float q = (ddp >= 0) ? (ddp + sqrtVal) : (ddp - sqrtVal);
		// we don't bother testing for division by zero
		const float t1 = q;
		const float t2 = (deltapdot - radius * radius) / q;
    result.x = std::min(t1,t2);
    result.y = std::max(t1,t2);
  }
  
  return result;
}

void TestClass::kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY) 
{
  const float x = float(tidX)*m_widthInv;
  const float y = float(tidY)*m_heightInv;
  *(rayPosAndNear) = float4(x, y, -1.0f, 0.0f);
  *(rayDirAndFar ) = float4(0, 0, 1, FLT_MAX);
  *flags           = 0;
}

void TestClass::kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, int* out_hit, uint tidX, uint tidY)
{
  CRT_Hit hit = m_pRayTraceImpl->RayQuery_NearestHit(*rayPosAndNear, *rayDirAndFar); 
  *out_hit = hit.primId;
}

void TestClass::kernel_TestColor(const int* in_hit, uint* out_color, uint tidX, uint tidY)
{
  if(*in_hit != -1)
    out_color[pitchOffset(tidX,tidY)] = 0x0000FFFF;
  else
    out_color[pitchOffset(tidX,tidY)] = 0x00FF0000;
}


void TestClass::BFRT_ReadAndCompute(uint tidX, uint tidY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  uint   flags;
  int    hit;

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

void TestClass::InitScene(int numBoxes, int numTris)
{
  boxes.resize(numBoxes*2);
  spheres.resize(numBoxes/2);

  trivets.resize(numTris*3);
  indices.resize(numTris*3);

  int boxesInString   = std::max(numBoxes/5, 5);
  int spheresInString = boxesInString/2;
  int trisInString    = std::max(numTris/5, 5);

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

  // (2) first quater of the screen contain spheres
  //
  for(int i=0;i<spheres.size();i++)
  { 
    int centerX = i%spheresInString;
    int centerY = i/spheresInString;
    spheres[i] = float4(0.5f*float(centerX + 0.5f)/float(spheresInString) + 0.5f, 0.5f*float(centerY + 0.5f)/float(spheresInString) + 0.5f, 1.0f*i + 0.0f, 0.02f);
  }

  // (3) anoher quater of the screen contain triangles
  //
  for(int i=0;i<numTris;i++)
  {
    const int centerX = i%trisInString;
    const int centerY = (numBoxes/boxesInString) + (i)/trisInString;

    trivets[i*3+0] = float4(float(centerX + 0.75f)/float(boxesInString) - boxSize, float(centerY + 0.75f)/float(boxesInString) - boxSize, i, 0.0f);
    trivets[i*3+1] = float4(float(centerX + 0.25f)/float(boxesInString),           float(centerY + 0.75f)/float(boxesInString), i, 0.0f);
    trivets[i*3+2] = float4(float(centerX + 0.75f)/float(boxesInString) - boxSize, float(centerY + 0.75f)/float(boxesInString) + boxSize, i, 0.0f);

    indices[i*3+0] = i*3+0;
    indices[i*3+1] = i*3+1;
    indices[i*3+2] = i*3+2;
  }
  
  auto bfRayTrace     = std::make_shared<BFRayTrace>(); 
  bfRayTrace->boxes   = boxes;
  bfRayTrace->spheres = spheres;
  bfRayTrace->trivets = trivets;
  bfRayTrace->indices = indices;

  m_pRayTraceImpl     = bfRayTrace;  
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

CRT_Hit BFRayTrace::RayQuery_NearestHit(float4 rayPosAndNear, float4 rayDirAndFar)
{
  const float3 rayPos    = to_float3(rayPosAndNear);
  const float3 rayDirInv = 1.0f/to_float3(rayDirAndFar);
  const float3 rayDir    = to_float3(rayDirAndFar);
  
  const float tNear = rayPosAndNear.w;
  const float tFar  = rayDirAndFar.w;
  
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

  for(uint32_t sphereId = 0; sphereId < spheres.size(); sphereId++) 
  {
    const float2 tm0 = RaySphereHit(rayPos, rayDir, spheres[sphereId]);
    const bool hit   = (tm0.x < tm0.y) && (tm0.y > tNear) && (tm0.x < tFar);
    if(hit)
      hitId = int(sphereId);
  }

  for (uint32_t triId = 0; triId < trivets.size(); triId+=3)
  {
    const float3 A_pos = to_float3(trivets[triId + 0]);
    const float3 B_pos = to_float3(trivets[triId + 1]);
    const float3 C_pos = to_float3(trivets[triId + 2]);
  
    const float3 edge1 = B_pos - A_pos;
    const float3 edge2 = C_pos - A_pos;
    const float3 pvec = cross(rayDir, edge2);
    const float3 tvec = rayPos - A_pos;
    const float3 qvec = cross(tvec, edge1);
  
    const float invDet = 1.0f / dot(edge1, pvec);
    const float v = dot(tvec, pvec) * invDet;
    const float u = dot(qvec, rayDir) * invDet;
    const float t = dot(edge2, qvec) * invDet;
  
    if (v >= -1e-6f && u >= -1e-6f && (u + v <= 1.0f + 1e-6f) && t > tNear && t < tFar)
      hitId = int(triId);
  }
  
  CRT_Hit hit;
  hit.primId = hitId;
  return hit;
}
