#include <vector>
#include <chrono>
#include <cfloat>

#include "test_class.h"

TestClass::TestClass(int w, int h) 
{
  m_widthInv  = 1.0f/float(w); 
  m_heightInv = 1.0f/float(h); 
  m_pRayTraceImpl = std::make_shared<BFRayTrace>();
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
    spheres[i] = float4(0.5f*float(centerX + 0.5f)/float(spheresInString) + 0.5f, 0.5f*float(centerY + 0.5f)/float(spheresInString) + 0.5f, 1.0f*i + 0.0f, 0.02f); // radius = 0.02f
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

  std::vector<CRT_AABB> boxesOnTopOfSpheres(spheres.size());
  for(size_t i=0;i<boxesOnTopOfSpheres.size();i++) 
  {
    boxesOnTopOfSpheres[i].boxMin.x = spheres[i].x - spheres[i].w;
    boxesOnTopOfSpheres[i].boxMin.y = spheres[i].y - spheres[i].w;
    boxesOnTopOfSpheres[i].boxMin.z = spheres[i].z - spheres[i].w; 

    boxesOnTopOfSpheres[i].boxMax.x = spheres[i].x + spheres[i].w;
    boxesOnTopOfSpheres[i].boxMax.y = spheres[i].y + spheres[i].w;
    boxesOnTopOfSpheres[i].boxMax.z = spheres[i].z + spheres[i].w; 
  }
  
  // put all geometry inaside impl.
  //
  m_pRayTraceImpl->ClearGeom();
  auto geomId2 = m_pRayTraceImpl->AddGeom_AABB(AbtractPrimitive::TAG_SPHERES, boxesOnTopOfSpheres.data(), boxesOnTopOfSpheres.size());
  auto geomId0 = m_pRayTraceImpl->AddGeom_Triangles3f((const float*)trivets.data(), trivets.size(), indices.data(), indices.size(), 0, 16);
  auto geomId1 = m_pRayTraceImpl->AddGeom_AABB(AbtractPrimitive::TAG_BOXES, (const CRT_AABB*)boxes.data(), numBoxes);

  m_pRayTraceImpl->ClearScene();
  m_pRayTraceImpl->AddInstance(geomId1, LiteMath::float4x4());
  m_pRayTraceImpl->AddInstance(geomId2, LiteMath::float4x4());
  m_pRayTraceImpl->AddInstance(geomId0, LiteMath::float4x4());
  m_pRayTraceImpl->CommitScene();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

uint32_t BFRayTrace::AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber, uint32_t a_flags, size_t vByteStride)
{
  trivets.resize(a_vertNumber);
  for(size_t i=0;i<a_vertNumber;i++)
  {
    trivets[i].x = a_vpos3f[i*(vByteStride/4)+0];
    trivets[i].y = a_vpos3f[i*(vByteStride/4)+1];
    trivets[i].z = a_vpos3f[i*(vByteStride/4)+2];
    trivets[i].w = 1.0f;
  }  
  
  indices = std::vector<uint32_t>(a_triIndices, a_triIndices + a_indNumber);

  const size_t oldSize = primitives.size();
  primitives.resize(oldSize + a_indNumber/3);
  for(size_t i = oldSize; i < primitives.size(); i++) 
  {
    const size_t oldIndex = i - oldSize;
    const uint32_t A      = a_triIndices[oldIndex*3+0];
    const uint32_t B      = a_triIndices[oldIndex*3+1];
    const uint32_t C      = a_triIndices[oldIndex*3+2];
    primitives[i] = new TrianglePrim(trivets[A], trivets[B], trivets[C], oldIndex); 
  }

  return 0;
}

uint32_t BFRayTrace::AddGeom_AABB(uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, uint32_t a_buildFlags, void** a_customPrimPtrs, size_t a_customPrimCount)
{
  const size_t oldSize = primitives.size();
  primitives.resize(oldSize + a_boxNumber);
  if(a_typeId == AbtractPrimitive::TAG_BOXES) 
  {
    for(size_t i = oldSize; i < primitives.size(); i++)
      primitives[i] = new AABBPrim(boxMinMaxF8[i-oldSize].boxMin, boxMinMaxF8[i-oldSize].boxMax, uint32_t(i-oldSize)); 
  }
  else if(a_typeId == AbtractPrimitive::TAG_SPHERES)
  {
    for(size_t i = oldSize; i < primitives.size(); i++)
      primitives[i] = new SpherePrim(boxMinMaxF8[i-oldSize].boxMin, boxMinMaxF8[i-oldSize].boxMax, uint32_t(i-oldSize)); 
  }
  else 
  {
    for(size_t i = oldSize; i < primitives.size(); i++)
      primitives[i] = new EmptyPrim(); 
  }
  return 0;
}

BFRayTrace::~BFRayTrace()
{
  for(size_t i=0;i<primitives.size();i++) {
    delete primitives[i];
    primitives[i] = nullptr;
  }
}
             
void BFRayTrace::UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber, uint32_t a_flags, size_t vByteStride) {}
void BFRayTrace::UpdateGeom_AABB(uint32_t a_geomId, uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, uint32_t a_buildFlags, void** a_customPrimPtrs, size_t a_customPrimCount) { }

CRT_Hit BFRayTrace::RayQuery_NearestHit(float4 rayPosAndNear, float4 rayDirAndFar)
{
  CRT_Hit hit;
  hit.primId = -1;
  
  CRT_LeafInfo info;
  info.primId = 0;
  info.aabbId = 0; 
  info.primId = 0; 
  info.instId = 0; 
  info.geomId = 0; 
  info.rayxId = 0; 
  info.rayyId = 0; 

  for(uint32_t primid = 0; primid < primitives.size(); primid++)
    primitives[primid]->Intersect(rayPosAndNear, rayDirAndFar, info, &hit, this); 

  return hit;
}
