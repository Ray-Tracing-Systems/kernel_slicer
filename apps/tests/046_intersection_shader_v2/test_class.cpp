#include <vector>
#include <chrono>
#include <cfloat>

#include "test_class.h"


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BFRayTrace::~BFRayTrace() {}

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
   
  const size_t oldSize     = m_primtable.size();
  const size_t oldTrisSize = m_tris.size();
  
  m_primtable.resize(oldSize + a_indNumber/3);

  std::vector<CRT_AABB> boxesOfTris(a_indNumber/3);

  for(size_t i = oldSize; i < m_primtable.size(); i++) 
  {
    const size_t oldIndex = i - oldSize;
    const uint32_t A      = a_triIndices[oldIndex*3+0];
    const uint32_t B      = a_triIndices[oldIndex*3+1];
    const uint32_t C      = a_triIndices[oldIndex*3+2];
    boxesOfTris[oldIndex].boxMin = LiteMath::min(trivets[A], LiteMath::min(trivets[B], trivets[C]));
    boxesOfTris[oldIndex].boxMax = LiteMath::max(trivets[A], LiteMath::max(trivets[B], trivets[C]));
    
    m_primtable[i] = ((TAG_TRIANGLES << 28) & 0xF0000000) | (uint32_t(m_tris.size()) & 0x0FFFFFFF);
    TrianglePrim tri;
    tri.m_primId = oldIndex;
    m_tris.push_back(tri);
  }
  
  const size_t oldBoxSize = allBoxes.size();
  allBoxes.insert(allBoxes.end(), boxesOfTris.begin(), boxesOfTris.end());
  
  for(size_t i = oldBoxSize; i<allBoxes.size(); i++) 
  {
    uint32_t primIndex   = uint32_t(oldSize + (i - oldBoxSize));
    allBoxes[i].boxMin.w = LiteMath::as_float(primIndex);
    allBoxes[i].boxMax.w = LiteMath::as_float(TAG_TRIANGLES);
  }

  BLASInfo info;
  info.startPrim = uint32_t(oldSize);
  info.sizePrims = uint32_t(m_primtable.size() - oldSize);
  info.startAABB = uint32_t(oldBoxSize);
  info.sizeAABBs = uint32_t(allBoxes.size() - oldBoxSize);

  startEnd.push_back(info); // may save TAG_TRIANGLES

  return uint32_t(startEnd.size() - 1);
}

uint32_t BFRayTrace::AddGeom_AABB(uint32_t a_typeId, const CRT_AABB* boxMinMaxF8, size_t a_boxNumber, void** a_customPrimPtrs, size_t a_customPrimCount)
{
  const size_t oldSize    = m_primtable.size();
  const size_t oldBoxSize = allBoxes.size();
  size_t actualPrimsCount = a_boxNumber;

  if(a_typeId == TAG_BOXES) 
  {
    const size_t oldBoxSize = m_aabbs.size();

    m_primtable.resize(oldSize + a_boxNumber);
    for(size_t i = oldSize; i < m_primtable.size(); i++) 
    {
      m_primtable[i] = ((TAG_BOXES << 28) & 0xF0000000) | (uint32_t(m_aabbs.size()) & 0x0FFFFFFF);
      AABBPrim box;
      box.boxMin = boxMinMaxF8[i-oldSize].boxMin;
      box.boxMax = boxMinMaxF8[i-oldSize].boxMax;
      m_aabbs.push_back(box);
    }
  }
  else if(a_typeId == TAG_SPHERES)
  {
    if(a_customPrimPtrs != nullptr)
    {
      actualPrimsCount = a_customPrimCount;
      m_primtable.resize(oldSize + a_customPrimCount);
      for(size_t i = oldSize; i < m_primtable.size(); i++) 
      {
        SpherePrim* pSphere = (SpherePrim*)(a_customPrimPtrs[i - oldSize]);

        m_primtable[i] = ((TAG_SPHERES << 28) & 0xF0000000) | (uint32_t(m_spheres.size()) & 0x0FFFFFFF);
        m_spheres.push_back(*pSphere);
      }
    }
    else
    {
      m_primtable.resize(oldSize + a_boxNumber);
      for(size_t i = oldSize; i < m_primtable.size(); i++) 
      {
        float4 center = 0.5f*(boxMinMaxF8[i-oldSize].boxMin + boxMinMaxF8[i-oldSize].boxMax);
        center.w      = 0.5f*(boxMinMaxF8[i-oldSize].boxMax.x - boxMinMaxF8[i-oldSize].boxMin.x);
      
        m_primtable[i] = ((TAG_SPHERES << 28) & 0xF0000000) | (uint32_t(m_spheres.size()) & 0x0FFFFFFF);
        SpherePrim sphere;
        sphere.sphData = center;
        m_spheres.push_back(sphere);
      }
    }
  }
  else 
  {
    m_primtable.resize(oldSize + a_boxNumber);
    for(size_t i = oldSize; i < m_primtable.size(); i++)
    {
      const uint32_t oldIndex = uint32_t(i-oldSize);
      m_primtable[i] = ((TAG_EMPTY << 28) & 0xF0000000) | (oldIndex & 0x0FFFFFFF);
    }
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

uint32_t BFRayTrace::IntersectionShader(float4 rayPosAndNear, float4 rayDirAndFar, CRT_LeafInfo info, CRT_Hit* pHit)
{
  const uint32_t tab = m_primtable[info.primId];
  const uint32_t tag = (tab & 0xF0000000) >> 28;
  const uint32_t pid = (tab & 0x0FFFFFFF);

  uint32_t res = TAG_EMPTY;
  switch(tag)
  {
    case TAG_BOXES:
    {
      const float3 rayDirInv = 1.0f/to_float3(rayDirAndFar);
      const float4 myBoxMin  = m_aabbs[pid].boxMin;
      const float4 myBoxMax  = m_aabbs[pid].boxMax;
      const float2 tMinMax   = RayBoxIntersection2( to_float3(rayPosAndNear), rayDirInv, to_float3(myBoxMin), to_float3(myBoxMax));
      
      if(tMinMax.x <= tMinMax.y && tMinMax.y >= rayPosAndNear.w && tMinMax.x <= rayDirAndFar.w)
      {
        pHit->t      = tMinMax.x;
        pHit->primId = pid; // m_aabbs[pid].m_primId; 
        pHit->geomId = info.geomId;
        pHit->instId = info.instId;   
        res = TAG_BOXES; 
      }
      else
        res = TAG_EMPTY;
    }
    break;
    
    case TAG_TRIANGLES:
    {
      const float3 rayPos = to_float3(rayPosAndNear);
      const float3 rayDir = to_float3(rayDirAndFar);
  
      const uint32_t A = indices[m_tris[pid].m_primId*3+0];
      const uint32_t B = indices[m_tris[pid].m_primId*3+1];
      const uint32_t C = indices[m_tris[pid].m_primId*3+2];
   
      const float3 A_pos = to_float3(trivets[A]);
      const float3 B_pos = to_float3(trivets[B]);
      const float3 C_pos = to_float3(trivets[C]);
    
      const float3 edge1 = B_pos - A_pos;
      const float3 edge2 = C_pos - A_pos;
      const float3 pvec = cross(rayDir, edge2);
      const float3 tvec = rayPos - A_pos;
      const float3 qvec = cross(tvec, edge1);
    
      const float invDet = 1.0f / dot(edge1, pvec);
      const float v = dot(tvec, pvec) * invDet;
      const float u = dot(qvec, rayDir) * invDet;
      const float t = dot(edge2, qvec) * invDet;
    
      if (v >= -1e-6f && u >= -1e-6f && (u + v <= 1.0f + 1e-6f) && t > rayPosAndNear.w && t < rayDirAndFar.w)
      {
        pHit->t      = t;
        pHit->primId = int(m_tris[pid].m_primId);
        pHit->geomId = info.geomId;
        pHit->instId = info.instId;
        res = TAG_TRIANGLES; 
      }
      else
        res = TAG_EMPTY;
    }
    break;

    case TAG_SPHERES:
    {
      const float2 tm0 = RaySphereHit(to_float3(rayPosAndNear), to_float3(rayDirAndFar), m_spheres[pid].sphData);
      const bool hit   = (tm0.x < tm0.y) && (tm0.y > rayPosAndNear.w) && (tm0.x < rayDirAndFar.w);
      if(hit)
      {
        pHit->t      = tm0.x;
        pHit->primId = info.primId;
        pHit->geomId = info.geomId;
        pHit->instId = info.instId;
        res = TAG_SPHERES; 
      }
      else
        res = TAG_EMPTY;
    }
    break;

    default:
    break;
  };
  
  return res;
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
        auto res = IntersectionShader(rayPosAndNear2, rayDirAndFar2, info, &hit); 
        if(res != TAG_EMPTY)
          break;
      }
    }
  }

  return hit;
}

void BFRayTrace::CommitScene(uint32_t options) 
{
  std::cout << "[BFRayTrace::CommitScene]: " << std::endl;
  for(size_t primId = 0; primId < m_primtable.size(); primId++) 
  {
    const uint32_t tab = m_primtable[primId];
    const uint32_t tag = (tab & 0xF0000000) >> 28;
    const uint32_t pid = (tab & 0x0FFFFFFF);
    std::cout << "(tag, primId,pid) = (" << tag << "," << primId << ", " << pid << ")" << std::endl; 
  
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TestClass::TestClass(int w, int h) 
{
  m_widthInv  = 1.0f/float(w); 
  m_heightInv = 1.0f/float(h); 
  m_pRayTraceImpl = std::make_shared<BFRayTrace>();
}

void TestClass::kernel2D_Render(uint a_sizeX, uint a_sizeY, uint* out_color)
{
  #ifndef _DEBUG
  #pragma omp parallel for collapse(2)
  #endif
  for(uint y=0;y<a_sizeY;y++)
  {
    for(uint x=0;x<a_sizeX;x++) 
    {
      const float xNorm = float(x)*m_widthInv;
      const float yNorm = float(y)*m_heightInv;
      const float4 rayPosAndNear = float4(xNorm, yNorm, -1.0f, 0.0f);
      const float4 rayDirAndFar  = float4(0, 0, 1, FLT_MAX);
      
      const auto hit = m_pRayTraceImpl->RayQuery_NearestHit(rayPosAndNear, rayDirAndFar);

      if(hit.primId != -1) 
        out_color[pitchOffset(x,y)] = palette[hit.primId % palette.size()];
      else
        out_color[pitchOffset(x,y)] = 0x00FF0000;
    }
  }
}

void TestClass::Render(uint a_sizeX, uint a_sizeY, uint* out_color)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_Render(a_sizeX, a_sizeY, out_color);
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
  
  // 2 separate spheres inside single geom object
  //
  SpherePrim singleSphere, sphere1, sphere2;
  {
    singleSphere.sphData = sphereCenter;
    sphere1.sphData      = float4(-0.1,0,0,0.03f);
    sphere2.sphData      = float4(+0.1,0.0,0,0.04f);
  }

  std::vector<CRT_AABB> sphereBoxes(4);
  {
    // sphere #1
    //
    sphereBoxes[0].boxMin.x = sphere1.sphData.x - sphere1.sphData.w;
    sphereBoxes[0].boxMin.y = sphere1.sphData.y - sphere1.sphData.w;
    sphereBoxes[0].boxMin.z = sphere1.sphData.z - sphere1.sphData.w; 

    sphereBoxes[0].boxMax.x = sphere1.sphData.x;
    sphereBoxes[0].boxMax.y = sphere1.sphData.y;
    sphereBoxes[0].boxMax.z = sphere1.sphData.z; 

    sphereBoxes[1].boxMin.x = sphere1.sphData.x;
    sphereBoxes[1].boxMin.y = sphere1.sphData.y;
    sphereBoxes[1].boxMin.z = sphere1.sphData.z; 

    sphereBoxes[1].boxMax.x = sphere1.sphData.x + sphere1.sphData.w;
    sphereBoxes[1].boxMax.y = sphere1.sphData.y + sphere1.sphData.w;
    sphereBoxes[1].boxMax.z = sphere1.sphData.z + sphere1.sphData.w;
    
    // sphere #2
    //
    sphereBoxes[2].boxMin.x = sphere2.sphData.x - sphere2.sphData.w;
    sphereBoxes[2].boxMin.y = sphere2.sphData.y - sphere2.sphData.w;
    sphereBoxes[2].boxMin.z = sphere2.sphData.z - sphere2.sphData.w; 

    sphereBoxes[2].boxMax.x = sphere2.sphData.x;
    sphereBoxes[2].boxMax.y = sphere2.sphData.y;
    sphereBoxes[2].boxMax.z = sphere2.sphData.z; 

    sphereBoxes[3].boxMin.x = sphere2.sphData.x;
    sphereBoxes[3].boxMin.y = sphere2.sphData.y;
    sphereBoxes[3].boxMin.z = sphere2.sphData.z; 

    sphereBoxes[3].boxMax.x = sphere2.sphData.x + sphere2.sphData.w;
    sphereBoxes[3].boxMax.y = sphere2.sphData.y + sphere2.sphData.w;
    sphereBoxes[3].boxMax.z = sphere2.sphData.z + sphere2.sphData.w; 
  }

  // put all geometry inaside impl.
  //
  m_pRayTraceImpl->ClearGeom();
  auto geomId2 = m_pRayTraceImpl->AddGeom_AABB(TAG_SPHERES, boxesOnTopOfSpheres.data(), boxesOnTopOfSpheres.size());
  auto geomId0 = m_pRayTraceImpl->AddGeom_Triangles3f((const float*)trivets.data(), trivets.size(), indices.data(), indices.size(), 0, 16);
  auto geomId1 = m_pRayTraceImpl->AddGeom_AABB(TAG_BOXES, (const CRT_AABB*)boxes.data(), numBoxes);

  void* spherePtr = (void*)&singleSphere; 
  auto geomId3    = m_pRayTraceImpl->AddGeom_AABB(TAG_SPHERES, (const CRT_AABB*)singleSphereBoxes.data(), singleSphereBoxes.size(), &spherePtr, 1);

  void* spheresPtrArray[] = {&sphere1, &sphere2}; 
  auto geomId4 = m_pRayTraceImpl->AddGeom_AABB(TAG_SPHERES, (const CRT_AABB*)sphereBoxes.data(), sphereBoxes.size(), spheresPtrArray, 2);

  float4x4 transformTris1 = LiteMath::translate4x4(float3(0.3f, 0.60f, 0.0f)) * LiteMath::rotate4x4Z(+LiteMath::DEG_TO_RAD*45.0f);
  float4x4 transformTris2 = LiteMath::translate4x4(float3(0.7f, 0.75f, 0.0f)) * LiteMath::rotate4x4Z(-LiteMath::DEG_TO_RAD*45.0f);

  float4x4 transformSpheres1 = LiteMath::translate4x4(float3(0.2f, 0.2f, 0.0f)) * LiteMath::rotate4x4Z(+LiteMath::DEG_TO_RAD*25.0f);
  float4x4 transformSpheres2 = LiteMath::translate4x4(float3(0.5f, 0.2f, 0.0f)) * LiteMath::rotate4x4Z(-LiteMath::DEG_TO_RAD*25.0f);
  float4x4 transformSpheres3 = LiteMath::translate4x4(float3(0.75f, 0.25f, 0.0f)) * LiteMath::rotate4x4Z(-LiteMath::DEG_TO_RAD*30.0f);

  m_pRayTraceImpl->ClearScene();
  
  m_pRayTraceImpl->AddInstance(geomId2, transformSpheres1);    // spheres
  m_pRayTraceImpl->AddInstance(geomId1, LiteMath::float4x4()); // boxes
  m_pRayTraceImpl->AddInstance(geomId3, transformSpheres2);    // spheres
  
  m_pRayTraceImpl->AddInstance(geomId1, LiteMath::translate4x4(float3(0.4f, 0.4f, 0.0f)));   // boxes
  m_pRayTraceImpl->AddInstance(geomId2, transformSpheres3); // spheres
  m_pRayTraceImpl->AddInstance(geomId0, transformTris1);    // triangles
  m_pRayTraceImpl->AddInstance(geomId4, transformTris2);    // spheres
  
  m_pRayTraceImpl->CommitScene();

  palette = { 0xffe6194b, 0xff3cb44b, 0xffffe119, 0xff0082c8,
              0xfff58231, 0xff911eb4, 0xff46f0f0, 0xfff032e6,
              0xffd2f53c, 0xfffabebe, 0xff008080, 0xffe6beff,
              0xffaa6e28, 0xfffffac8, 0xff800000, 0xffaaffc3,
              0xff808000, 0xffffd8b1, 0xff000080, 0xff808080 };
}