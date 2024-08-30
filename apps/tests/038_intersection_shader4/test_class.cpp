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

void TestClass::kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, CRT_Hit* out_hit, uint tidX, uint tidY)
{ 
  *out_hit = m_pRayTraceImpl->RayQuery_NearestHit(*rayPosAndNear, *rayDirAndFar); 
}

void TestClass::kernel_TestColor(const CRT_Hit* in_hit, uint* out_color, uint tidX, uint tidY)
{
  if(in_hit->primId != -1) 
  {
    out_color[pitchOffset(tidX,tidY)] = palette[in_hit->primId % palette.size()];
  }
  else
    out_color[pitchOffset(tidX,tidY)] = 0x00FF0000;
}


void TestClass::Render(uint tidX, uint tidY, uint* out_color)
{
  float4  rayPosAndNear, rayDirAndFar;
  uint    flags;
  CRT_Hit hit;

  kernel_InitEyeRay(&flags, &rayPosAndNear, &rayDirAndFar, tidX, tidY);
  kernel_RayTrace  (&rayPosAndNear, &rayDirAndFar, &hit, tidX, tidY);
  kernel_TestColor (&hit, out_color, tidX, tidY);
}

void TestClass::RenderBlock(uint tidX, uint tidY, uint* out_color, uint32_t a_numPasses)
{
  auto before = std::chrono::high_resolution_clock::now();
  #ifndef _DEBUG
  #pragma omp parallel for collapse(2)
  #endif
  for(int y=0;y<tidY;y++)
    for(int x=0;x<tidX;x++)
      for(int p=0;p<a_numPasses;p++)
        Render(x,y,out_color);
  m_time1 = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName).find("Render") != std::string::npos)
    a_out[0] = m_time1;
}

void TestClass::InitScene(int numBoxes, int numTris)
{
  boxes.resize(numBoxes*2);
  spheres.resize(numBoxes/2);

  trivets.resize(numTris*3);
  indices.resize(numTris*3);

  int boxesInString   = int(std::sqrt(numBoxes)); // std::max(numBoxes/5, 5);
  int trisInString    = int(std::sqrt(numTris));

  float boxSize = 0.05f/float(boxesInString);
  float triSize = 0.15f/float(trisInString);

  // (1) first half of the screen contain boxes
  //
  for(int i=0;i<numBoxes;i++)
  { 
    int centerX = i%boxesInString;
    int centerY = i/boxesInString;
    boxes[i*2+0] = float4(float(centerX + 0.5f)/float(20) - boxSize, float(centerY + 0.5f)/float(20) - boxSize, 1.0f*i + 0.0f, 0.0f);
    boxes[i*2+1] = float4(float(centerX + 0.5f)/float(20) + boxSize, float(centerY + 0.5f)/float(20) + boxSize, 1.0f*i + 0.5f, 0.0f);
  }

  // (2) first quater of the screen contain spheres
  //
  spheres.resize(9);
  for(int y=0;y<3;y++) 
  {
    for(int x=0;x<3;x++)
    { 
      int centerX = x;
      int centerY = y;
      spheres[y*3+x] = float4(float(centerX)/float(16), float(centerY)/float(16), 1.0f, 0.02f); // radius = 0.02f
    }
  }

  // (3) anoher quater of the screen contain triangles
  //
  for(int i=0;i<numTris;i++)
  {
    const int centerX = i%trisInString;
    const int centerY = i/trisInString;

    trivets[i*3+0] = float4(float(centerX)/float(12) - triSize, float(centerY)/float(12) - triSize, i, 0.0f);
    trivets[i*3+1] = float4(float(centerX)/float(12),           float(centerY)/float(12), i, 0.0f);
    trivets[i*3+2] = float4(float(centerX)/float(12) - triSize, float(centerY)/float(12) + triSize, i, 0.0f);

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

  // single sphere with several bounding boxes
  //
  float4 sphereCenter(0,0,0,0.05f);
  std::vector<CRT_AABB> singleSphereBoxes(3);
  {
    singleSphereBoxes[0].boxMin.x = sphereCenter.x - sphereCenter.w;
    singleSphereBoxes[0].boxMin.y = sphereCenter.y - sphereCenter.w;
    singleSphereBoxes[0].boxMin.z = sphereCenter.z - sphereCenter.w; 

    singleSphereBoxes[0].boxMax.x = sphereCenter.x;
    singleSphereBoxes[0].boxMax.y = sphereCenter.y;
    singleSphereBoxes[0].boxMax.z = sphereCenter.z; 

    singleSphereBoxes[1].boxMin.x = sphereCenter.x;
    singleSphereBoxes[1].boxMin.y = sphereCenter.y;
    singleSphereBoxes[1].boxMin.z = sphereCenter.z; 

    singleSphereBoxes[1].boxMax.x = sphereCenter.x + sphereCenter.w;
    singleSphereBoxes[1].boxMax.y = sphereCenter.y + sphereCenter.w;
    singleSphereBoxes[1].boxMax.z = sphereCenter.z + sphereCenter.w; 

    singleSphereBoxes[2].boxMin.x = sphereCenter.x;
    singleSphereBoxes[2].boxMin.y = sphereCenter.y - sphereCenter.w;
    singleSphereBoxes[2].boxMin.z = sphereCenter.z - sphereCenter.w; 

    singleSphereBoxes[2].boxMax.x = sphereCenter.x +  sphereCenter.w;
    singleSphereBoxes[2].boxMax.y = sphereCenter.y;
    singleSphereBoxes[2].boxMax.z = sphereCenter.z; 
  }
   
  SpherePrim* pSingleSphere = new SpherePrim(sphereCenter, 0);
  
  // 2 separate spheres inside single geom object
  //
  float4 sphereData1(-0.1,0,0,0.03f);
  float4 sphereData2(+0.1,0.0,0,0.04f);
  SpherePrim* pSphere1 = new SpherePrim(sphereData1, 0);
  SpherePrim* pSphere2 = new SpherePrim(sphereData2, 0);

  std::vector<CRT_AABB> sphereBoxes(4);
  {
    // sphere #1
    //
    sphereBoxes[0].boxMin.x = sphereData1.x - sphereData1.w;
    sphereBoxes[0].boxMin.y = sphereData1.y - sphereData1.w;
    sphereBoxes[0].boxMin.z = sphereData1.z - sphereData1.w; 

    sphereBoxes[0].boxMax.x = sphereData1.x;
    sphereBoxes[0].boxMax.y = sphereData1.y;
    sphereBoxes[0].boxMax.z = sphereData1.z; 

    sphereBoxes[1].boxMin.x = sphereData1.x;
    sphereBoxes[1].boxMin.y = sphereData1.y;
    sphereBoxes[1].boxMin.z = sphereData1.z; 

    sphereBoxes[1].boxMax.x = sphereData1.x + sphereData1.w;
    sphereBoxes[1].boxMax.y = sphereData1.y + sphereData1.w;
    sphereBoxes[1].boxMax.z = sphereData1.z + sphereData1.w;
    
    // sphere #2
    //
    sphereBoxes[2].boxMin.x = sphereData2.x - sphereData2.w;
    sphereBoxes[2].boxMin.y = sphereData2.y - sphereData2.w;
    sphereBoxes[2].boxMin.z = sphereData2.z - sphereData2.w; 

    sphereBoxes[2].boxMax.x = sphereData2.x;
    sphereBoxes[2].boxMax.y = sphereData2.y;
    sphereBoxes[2].boxMax.z = sphereData2.z; 

    sphereBoxes[3].boxMin.x = sphereData2.x;
    sphereBoxes[3].boxMin.y = sphereData2.y;
    sphereBoxes[3].boxMin.z = sphereData2.z; 

    sphereBoxes[3].boxMax.x = sphereData2.x + sphereData2.w;
    sphereBoxes[3].boxMax.y = sphereData2.y + sphereData2.w;
    sphereBoxes[3].boxMax.z = sphereData2.z + sphereData2.w; 
  }

  // put all geometry inaside impl.
  //
  m_pRayTraceImpl->ClearGeom();
  auto geomId2 = m_pRayTraceImpl->AddGeom_AABB(AbtractPrimitive::TAG_SPHERES, boxesOnTopOfSpheres.data(), boxesOnTopOfSpheres.size());
  auto geomId0 = m_pRayTraceImpl->AddGeom_Triangles3f((const float*)trivets.data(), trivets.size(), indices.data(), indices.size(), 0, 16);
  auto geomId1 = m_pRayTraceImpl->AddGeom_AABB(AbtractPrimitive::TAG_BOXES, (const CRT_AABB*)boxes.data(), numBoxes);

  void* spherePtr = (void*)pSingleSphere; 
  auto geomId3 = m_pRayTraceImpl->AddGeom_AABB(AbtractPrimitive::TAG_SPHERES, (const CRT_AABB*)singleSphereBoxes.data(), singleSphereBoxes.size(), &spherePtr, 1);

  void* spheresPtrArray[] = {(void*)pSphere1, (void*)pSphere2}; 
  auto geomId4 = m_pRayTraceImpl->AddGeom_AABB(AbtractPrimitive::TAG_SPHERES, (const CRT_AABB*)sphereBoxes.data(), sphereBoxes.size(), spheresPtrArray, 2);

  float4x4 transformTris1 = LiteMath::translate4x4(float3(0.3f, 0.60f, 0.0f)) * LiteMath::rotate4x4Z(+LiteMath::DEG_TO_RAD*45.0f);
  float4x4 transformTris2 = LiteMath::translate4x4(float3(0.7f, 0.75f, 0.0f)) * LiteMath::rotate4x4Z(-LiteMath::DEG_TO_RAD*45.0f);

  float4x4 transformSpheres1 = LiteMath::translate4x4(float3(0.2f, 0.2f, 0.0f)) * LiteMath::rotate4x4Z(+LiteMath::DEG_TO_RAD*25.0f);
  float4x4 transformSpheres2 = LiteMath::translate4x4(float3(0.5f, 0.2f, 0.0f)) * LiteMath::rotate4x4Z(-LiteMath::DEG_TO_RAD*25.0f);
  float4x4 transformSpheres3 = LiteMath::translate4x4(float3(0.75f, 0.25f, 0.0f)) * LiteMath::rotate4x4Z(-LiteMath::DEG_TO_RAD*30.0f);

  m_pRayTraceImpl->ClearScene();
  
  // boxes
  //
  m_pRayTraceImpl->AddInstance(geomId1, LiteMath::float4x4());
  m_pRayTraceImpl->AddInstance(geomId1, LiteMath::translate4x4(float3(0.4f, 0.4f, 0.0f)));
  
  // spheres
  //
  m_pRayTraceImpl->AddInstance(geomId2, transformSpheres1);
  m_pRayTraceImpl->AddInstance(geomId3, transformSpheres2);
  m_pRayTraceImpl->AddInstance(geomId2, transformSpheres3);
  m_pRayTraceImpl->AddInstance(geomId4, transformTris2);
  
  // triangles
  //
  m_pRayTraceImpl->AddInstance(geomId0, transformTris1);
  //m_pRayTraceImpl->AddInstance(geomId0, transformTris2);
  
  m_pRayTraceImpl->CommitScene();

  palette = { 0xffe6194b, 0xff3cb44b, 0xffffe119, 0xff0082c8,
              0xfff58231, 0xff911eb4, 0xff46f0f0, 0xfff032e6,
              0xffd2f53c, 0xfffabebe, 0xff008080, 0xffe6beff,
              0xffaa6e28, 0xfffffac8, 0xff800000, 0xffaaffc3,
              0xff808000, 0xffffd8b1, 0xff000080, 0xff808080 };
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
  std::vector<CRT_AABB> boxesOfTris(a_indNumber/3);

  for(size_t i = oldSize; i < primitives.size(); i++) 
  {
    const size_t oldIndex = i - oldSize;
    const uint32_t A      = a_triIndices[oldIndex*3+0];
    const uint32_t B      = a_triIndices[oldIndex*3+1];
    const uint32_t C      = a_triIndices[oldIndex*3+2];
    primitives [i] = new TrianglePrim(oldIndex); 
    boxesOfTris[oldIndex].boxMin = LiteMath::min(trivets[A], LiteMath::min(trivets[B], trivets[C]));
    boxesOfTris[oldIndex].boxMax = LiteMath::max(trivets[A], LiteMath::max(trivets[B], trivets[C]));
  }
  
  const size_t oldBoxSize = allBoxes.size();
  allBoxes.insert(allBoxes.end(), boxesOfTris.begin(), boxesOfTris.end());
  
  for(size_t i = oldBoxSize; i<allBoxes.size(); i++) 
  {
    uint32_t primIndex   = uint32_t(oldSize + (i - oldBoxSize));
    allBoxes[i].boxMin.w = LiteMath::as_float(primIndex);
    allBoxes[i].boxMax.w = LiteMath::as_float(AbtractPrimitive::TAG_TRIANGLES);
  }

  BLASInfo info;
  info.startPrim = uint32_t(oldSize);
  info.sizePrims = uint32_t(primitives.size() - oldSize);
  info.startAABB = uint32_t(oldBoxSize);
  info.sizeAABBs = uint32_t(allBoxes.size() - oldBoxSize);

  startEnd.push_back(info); // may save TAG_TRIANGLES

  return uint32_t(startEnd.size() - 1);
}

uint32_t BFRayTrace::AddGeom_AABB(uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, void** a_customPrimPtrs, size_t a_customPrimCount)
{
  const size_t oldSize    = primitives.size();
  const size_t oldBoxSize = allBoxes.size();
  size_t actualPrimsCount = a_boxNumber;
  if(a_typeId == AbtractPrimitive::TAG_BOXES) 
  {
    primitives.resize(oldSize + a_boxNumber);
    for(size_t i = oldSize; i < primitives.size(); i++)
      primitives[i] = new AABBPrim(boxMinMaxF8[i-oldSize].boxMin, boxMinMaxF8[i-oldSize].boxMax, uint32_t(i-oldSize)); 
  }
  else if(a_typeId == AbtractPrimitive::TAG_SPHERES)
  {
    if(a_customPrimPtrs != nullptr)
    {
      actualPrimsCount = a_customPrimCount;
      primitives.resize(oldSize + a_customPrimCount);
      for(size_t i = oldSize; i < primitives.size(); i++)
        primitives[i] = (SpherePrim*)(a_customPrimPtrs[i - oldSize]);
    }
    else
    {
      primitives.resize(oldSize + a_boxNumber);
      for(size_t i = oldSize; i < primitives.size(); i++) {
        float4 center = 0.5f*(boxMinMaxF8[i-oldSize].boxMin + boxMinMaxF8[i-oldSize].boxMax);
        center.w      = 0.5f*(boxMinMaxF8[i-oldSize].boxMax.x - boxMinMaxF8[i-oldSize].boxMin.x);
        primitives[i] = new SpherePrim(center, uint32_t(i-oldSize)); 
      }
    }

  }
  else 
  {
    primitives.resize(oldSize + a_boxNumber);
    for(size_t i = oldSize; i < primitives.size(); i++)
      primitives[i] = new EmptyPrim(); 
  }

  allBoxes.insert(allBoxes.end(), boxMinMaxF8, boxMinMaxF8 + a_boxNumber);
  
  const size_t div = a_boxNumber/actualPrimsCount;
  for(size_t i = oldBoxSize; i<allBoxes.size(); i++) 
  {
    uint32_t primIndex   = uint32_t(oldSize + (i - oldBoxSize) / div);
    allBoxes[i].boxMin.w = LiteMath::as_float(primIndex);
    allBoxes[i].boxMax.w = LiteMath::as_float(a_typeId);
  }

  BLASInfo info;
  {
    info.startPrim = uint32_t(oldSize);
    info.sizePrims = uint32_t(actualPrimsCount);
    info.startAABB = uint32_t(oldBoxSize);
    info.sizeAABBs = uint32_t(a_boxNumber);
  }
  startEnd.push_back(info); // may save a_typeId

  return uint32_t(startEnd.size() - 1);
}

uint32_t BFRayTrace::AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix)
{
  if((a_geomId & CRT_GEOM_MASK_AABB_BIT) != 0)
    a_geomId = (a_geomId & CRT_GEOM_MASK_AABB_BIT_RM);
  m_instStartEnd.push_back(startEnd[a_geomId]);
  m_instMatricesFwd.push_back(a_matrix);
  m_instMatricesInv.push_back(LiteMath::inverse4x4(a_matrix));
  return uint32_t(m_instStartEnd.size());
}

BFRayTrace::~BFRayTrace()
{
  for(size_t i=0;i<primitives.size();i++) {
    delete primitives[i];
    primitives[i] = nullptr;
  }
}
             
void BFRayTrace::UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber, uint32_t a_flags, size_t vByteStride) {}
void BFRayTrace::UpdateGeom_AABB(uint32_t a_geomId, uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, void** a_customPrimPtrs, size_t a_customPrimCount) { }

static inline float3 matmul3x3(float4x4 m, float3 v)
{ 
  return to_float3(m*to_float4(v, 0.0f));
}

static inline float3 matmul4x3(float4x4 m, float3 v)
{
  return to_float3(m*to_float4(v, 1.0f));
}

CRT_Hit BFRayTrace::RayQuery_NearestHit(float4 rayPosAndNear, float4 rayDirAndFar)
{
  CRT_Hit hit;
  hit.primId = -1;
  
  CRT_LeafInfo info;

  for(uint32_t instId = 0; instId < m_instStartEnd.size(); instId++) 
  {
    const auto  startEnd = m_instStartEnd[instId];
    const float3 ray_pos = matmul4x3(m_instMatricesInv[instId], to_float3(rayPosAndNear));
    const float3 ray_dir = matmul3x3(m_instMatricesInv[instId], to_float3(rayDirAndFar)); 
    
    const float4 rayPosAndNear2 = to_float4(ray_pos, rayPosAndNear.w);
    const float4 rayDirAndFar2  = to_float4(ray_dir, rayDirAndFar.w);

    info.instId = instId;
    info.geomId = instId; // TODO: get from some array

    const float3 rayDirInv = 1.0f/to_float3(rayDirAndFar2);

    // list all intersected boxes, for each box get primitive id and intersect primitive id
    //
    for(uint32_t boxId = startEnd.startAABB; boxId < startEnd.startAABB + startEnd.sizeAABBs; boxId++)
    {
      CRT_AABB currBox = allBoxes[boxId];
      uint32_t primid  = LiteMath::as_uint(currBox.boxMin.w); 
      float2 tMinMax   = RayBoxIntersection2( to_float3(rayPosAndNear2), rayDirInv, to_float3(currBox.boxMin), to_float3(currBox.boxMax) );
      if(tMinMax.x <= tMinMax.y && tMinMax.y >= rayPosAndNear2.w && tMinMax.x <= rayDirAndFar2.w)
      {  
        info.aabbId = boxId;
        info.primId = primid;
        auto res = primitives[primid]->Intersect(rayPosAndNear2, rayDirAndFar2, info, &hit, this); 
        if(res != AbtractPrimitive::TAG_EMPTY)
          break;
      }
    }

    //for(uint32_t primid = startEnd.startPrim; primid < startEnd.startPrim + startEnd.sizePrims; primid++) 
    //{
    //  info.aabbId = primid;
    //  info.geomId = primid; // TODO: use remap table to get it
    //  primitives[primid]->Intersect(rayPosAndNear2, rayDirAndFar2, info, &hit, this); 
    //}
  }

  return hit;
}
