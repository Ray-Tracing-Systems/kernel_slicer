#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>
#include <new>
#include <cfloat>

void TestClass::InitSceneMaterials(int a_numSpheres, int a_seed)
{ 
  // using in place new to create objects on the CPU side
  //
  m_materials.resize(11);
  
  new (m_materials.data() + 0 ) EmptyMaterial()                                      ;
  new (m_materials.data() + 1 ) LambertMaterial(float3(0.5,0.5,0.5))                 ;
  new (m_materials.data() + 2 ) LambertMaterial(float3(0.6,0.0235294,0.0235294))     ;
  new (m_materials.data() + 3 ) LambertMaterial(float3(0.0235294, 0.6, 0.0235294))   ;
  new (m_materials.data() + 4 ) GGXGlossyMaterial(float3(0.6,0.6,0.1))               ;
  new (m_materials.data() + 5 ) LambertMaterial(float3(0.0847059, 0.144706,0.265882));
  new (m_materials.data() + 6 ) PerfectMirrorMaterial                                ;
  new (m_materials.data() + 7 ) LambertMaterial(float3(0.25,0.0,0.5))                ;
  new (m_materials.data() + 8 ) PerfectMirrorMaterial                                ;
  new (m_materials.data() + 9 ) PerfectMirrorMaterial                                ;
  new (m_materials.data() + 10) EmissiveMaterial(20.0f)                              ;
  
  m_emissiveMaterialId = 10;
  
  std::cout << "[info]: sizeof(IMaterial)         = " << sizeof(IMaterial) << std::endl;
  std::cout << "[info]: sizeof(LambertMaterial)   = " << sizeof(LambertMaterial) << std::endl;
  std::cout << "[info]: sizeof(GGXGlossyMaterial) = " << sizeof(GGXGlossyMaterial) << std::endl;
  std::cout << "[info]: sizeof(MirrorMaterial)    = " << sizeof(PerfectMirrorMaterial) << std::endl;
  std::cout << "[info]: sizeof(EmissiveMaterial)  = " << sizeof(EmissiveMaterial) << std::endl;
  std::cout << "[info]: sizeof(EmptyMaterial)     = " << sizeof(EmptyMaterial) << std::endl;
  
  char* pBegin = (char*)m_materials.data();
  {
    char* p_m_color   = (char*)(&m_materials[0].m_color[0]);
    char* p_roughness = (char*)(&m_materials[0].roughness);
    char* p_m_tag     = (char*)(&m_materials[0].m_tag);

    auto  m_colorOffset   = static_cast<int64_t>(p_m_color   - pBegin);
    auto  roughnessOffset = static_cast<int64_t>(p_roughness - pBegin);
    auto  m_tagOffset     = static_cast<int64_t>(p_m_tag     - pBegin);

    std::cout << "[info]: offset_of(m_color)   = " << m_colorOffset   << std::endl;
    std::cout << "[info]: offset_of(roughness) = " << roughnessOffset << std::endl;
    std::cout << "[info]: offset_of(m_tag )    = " << m_tagOffset     << std::endl;
  }
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

  const float3 rayDir = EyeRayDir(float(x), float(y), float(WIN_WIDTH), float(WIN_HEIGHT), m_worldViewProjInv); 
  const float3 rayPos = camPos;
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
}

void TestClass::kernel_InitEyeRay2(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar,
                                                                   float4* accumColor,    float4* accumuThoroughput) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,0);

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  const float3 rayDir = EyeRayDir(float(x), float(y), float(WIN_WIDTH), float(WIN_HEIGHT), m_worldViewProjInv); 
  const float3 rayPos = camPos;
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
}

static float2 RayBoxIntersectionLite(const float3 ray_pos, const float3 ray_dir_inv, const float boxMin[3], const float boxMax[3])
{
  const float lo = ray_dir_inv.x*(boxMin[0] - ray_pos.x);
  const float hi = ray_dir_inv.x*(boxMax[0] - ray_pos.x);

  float tmin = std::min(lo, hi);
  float tmax = std::max(lo, hi);

  const float lo1 = ray_dir_inv.y*(boxMin[1] - ray_pos.y);
  const float hi1 = ray_dir_inv.y*(boxMax[1] - ray_pos.y);

  tmin = std::max(tmin, std::min(lo1, hi1));
  tmax = std::min(tmax, std::max(lo1, hi1));

  const float lo2 = ray_dir_inv.z*(boxMin[2] - ray_pos.z);
  const float hi2 = ray_dir_inv.z*(boxMax[2] - ray_pos.z);

  tmin = std::max(tmin, std::min(lo2, hi2));
  tmax = std::min(tmax, std::max(lo2, hi2));

  return make_float2(tmin, tmax); //(tmin <= tmax) && (tmax > 0.f);
}

static void IntersectAllPrimitivesInLeaf(const float4 rayPosAndNear, const float4 rayDirAndFar,
                                         __global const uint* a_indices, uint a_start, uint a_count, __global const float4* a_vert,
                                         Lite_Hit* pHit, float2* pBars)
{
  const uint triAddressEnd = a_start + a_count;
  for (uint triAddress = a_start; triAddress < triAddressEnd; triAddress = triAddress + 3u)
  {
    const uint A = a_indices[triAddress + 0];
    const uint B = a_indices[triAddress + 1];
    const uint C = a_indices[triAddress + 2];

    const float4 A_pos = a_vert[A];
    const float4 B_pos = a_vert[B];
    const float4 C_pos = a_vert[C];

    const float4 edge1 = B_pos - A_pos;
    const float4 edge2 = C_pos - A_pos;
    const float4 pvec  = cross3(rayDirAndFar, edge2);
    const float4 tvec  = rayPosAndNear - A_pos;
    const float4 qvec  = cross3(tvec, edge1);
    const float dotTmp = dot(to_float3(edge1), to_float3(pvec));
    const float invDet = 1.0f / (dotTmp > 1e-6f ? dotTmp : 1e-6f);

    const float v = dot(to_float3(tvec), to_float3(pvec))*invDet;
    const float u = dot(to_float3(qvec), to_float3(rayDirAndFar))*invDet;
    const float t = dot(to_float3(edge2), to_float3(qvec))*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > rayPosAndNear.w && t < pHit->t)
    {
      pHit->t      = t;
      pHit->primId = int(triAddress)/3;
      (*pBars)     = make_float2(u,v);
    }
  }

}

static inline float3 SafeInverse_4to3(float4 d)
{
  const float ooeps = 1.0e-36f; // Avoid div by zero.
  float3 res;
  res.x = 1.0f / (std::abs(d.x) > ooeps ? d.x : std::copysign(ooeps, d.x));
  res.y = 1.0f / (std::abs(d.y) > ooeps ? d.y : std::copysign(ooeps, d.y));
  res.z = 1.0f / (std::abs(d.z) > ooeps ? d.z : std::copysign(ooeps, d.z));
  return res;
}

bool TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                Lite_Hit* out_hit, float2* out_bars, uint* out_mid)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  const float3 rayDirInv = SafeInverse_4to3(rayDir);

  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;

  float2 baricentrics = float2(0,0);

  uint nodeIdx = 0;
  while(nodeIdx < 0xFFFFFFFE)
  {
    const BVHNode currNode = m_nodes[nodeIdx];
    const float2 boxHit           = RayBoxIntersectionLite(to_float3(rayPos), rayDirInv, currNode.boxMin, currNode.boxMax);
    const bool   intersects       = (boxHit.x <= boxHit.y) && (boxHit.y > rayPos.w) && (boxHit.x < res.t); // (tmin <= tmax) && (tmax > 0.f) && (tmin < curr_t)

    if(intersects && currNode.leftOffset == 0xFFFFFFFF) //leaf
    {
      Interval startCount = m_intervals[nodeIdx];
      IntersectAllPrimitivesInLeaf(rayPos, rayDir, m_indicesReordered.data(), startCount.start*3, startCount.count*3, m_vPos4f.data(), 
                                   &res, &baricentrics);
    }

    nodeIdx = (currNode.leftOffset == 0xFFFFFFFF || !intersects) ? currNode.escapeIndex : currNode.leftOffset;
    nodeIdx = (nodeIdx == 0) ? 0xFFFFFFFE : nodeIdx;
  }
  
  // intersect flat light under roof
  {
    const float tLightHit  = (m_lightGeom.boxMax.y - rayPos.y)*rayDirInv.y;
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
  
  uint mid = 0;
  if(res.primId != -1) 
    mid = (res.geomId == HIT_FLAT_LIGHT_GEOM) ? m_emissiveMaterialId : m_materialIds[res.primId]+1;

  *out_hit  = res;
  *out_bars = baricentrics;
  *out_mid  = mid;  

  return (res.primId != -1);
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_RealColorToUint32(uint tid, uint* mid, uint* out_color)
{
  const uint mid2 = *mid;
  const float3 color = (m_materials.data() + mid2)->GetColor();
  out_color[tid] = RealColorToUint32_f3(color);
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
  float2   baricentrics; 
  uint     mid;
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics, &mid))
    return;
    
  kernel_RealColorToUint32(tid, &mid, out_color);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*WIN_WIDTH+x] += *a_accumColor;
}

void  TestClass::kernel_NextBounce(uint tid, uint* mid, const Lite_Hit* in_hit, const float2* in_bars, 
                                   float4* rayPosAndNear, float4* rayDirAndFar, RandomGen* pGen, 
                                   float4* accumColor, float4* accumThoroughput)
  {
    const Lite_Hit lHit  = *in_hit;
    const float3 ray_dir = to_float3(*rayDirAndFar);
    
    SurfaceHit hit;
    hit.pos  = to_float3(*rayPosAndNear) + lHit.t*ray_dir;
    hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, *in_bars, m_indicesReordered.data(), m_vNorm4f.data()); 
  
    RandomGen gen   = pGen[tid];
    const float2 uv = rndFloat2_Pseudo(&gen);
    pGen[tid]       = gen;
    
    const uint       mid2     = *mid;
    const BxDFSample matSam   = (m_materials.data() + mid2)->SampleAndEvalBxDF(*rayPosAndNear, *rayDirAndFar, hit, uv);
    const float3     bxdfVal  = matSam.brdfVal * (1.0f / std::max(matSam.pdfVal, 1e-10f));
    const float      cosTheta = dot(matSam.newDir, hit.norm);
    
    if(matSam.flags != 0) 
    {
      *accumColor    = to_float4(matSam.brdfVal,0.0f)*(*accumThoroughput);
      *rayPosAndNear = make_float4(0,10000000.0f,0,0); // shoot ray out of scene
      *rayDirAndFar  = make_float4(0,1.0f,0,0);        // 
    }
    else
    {
      *rayPosAndNear    = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.newDir), 0.0f);
      *rayDirAndFar     = to_float4(matSam.newDir, FLT_MAX);
      *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f);
    }
  }

void TestClass::NaivePathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput);

  Lite_Hit hit; 
  float2   baricentrics; 

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    uint     mid;
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics, &mid))
      break;
    
    kernel_NextBounce(tid, &mid, &hit, &baricentrics, 
                      &rayPosAndNear, &rayDirAndFar, m_randomGens.data(), 
                      &accumColor, &accumThoroughput);
  }

  kernel_ContributeToImage(tid, &accumColor, in_pakedXY, 
                           out_color);
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
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    CastSingleRay(i, in_pakedXY, out_color);
}

void TestClass::NaivePathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passNum;j++)
      NaivePathTrace(i, 6, in_pakedXY, out_color);
  m_executionTimePT = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "NaivePathTrace" || std::string(a_funcName) == "NaivePathTraceBlock")
    a_out[0] = m_executionTimePT;
}