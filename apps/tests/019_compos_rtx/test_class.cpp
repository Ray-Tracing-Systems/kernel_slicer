#include <vector>
#include <chrono>
#include "test_class.h"

void TestClass::InitTris(size_t numTris, std::vector<float4>& trivets, std::vector<uint32_t>& indices)
{
  trivets.resize(numTris*3);
  indices.resize(numTris*3);

  int trisInString  = std::max(int(numTris)/10, 5);
  float triSize     = 0.25f/float(trisInString);

  for(int i=0;i<numTris;i++)
  {
    int centerX = i%trisInString;
    int centerY = (i)/trisInString;

    trivets[i*3+0] = float4(float(centerX + 0.75f)/float(trisInString) - triSize, float(centerY + 0.75f)/float(trisInString) - triSize, i, 0.0f);
    trivets[i*3+1] = float4(float(centerX + 0.25f)/float(trisInString),           float(centerY + 0.75f)/float(trisInString), i, 0.0f);
    trivets[i*3+2] = float4(float(centerX + 0.75f)/float(trisInString) - triSize, float(centerY + 0.75f)/float(trisInString) + triSize, i, 0.0f);

    indices[i*3+0] = i*3+0;
    indices[i*3+1] = i*3+1;
    indices[i*3+2] = i*3+2;
  }
}

TestClass::TestClass(int w, int h) 
{
  m_widthInv  = 1.0f/float(w); 
  m_heightInv = 1.0f/float(h); 
  //m_pAccelStruct = std::shared_ptr<ISceneObject>(CreateSceneRT(""), [](ISceneObject *p) { DeleteSceneRT(p); } );
  auto pBFImpl   = std::make_shared<BFRayTrace>();
  m_pAccelStruct = pBFImpl;
}

void TestClass::InitScene()
{
  std::vector<float4> vPos4f;
  std::vector<uint32_t> indices;
  this->InitTris(16, vPos4f, indices);

  m_pAccelStruct->ClearGeom();
  auto geomId = m_pAccelStruct->AddGeom_Triangles3f((const float*)vPos4f.data(), vPos4f.size(), indices.data(), indices.size(), BUILD_HIGH, sizeof(float)*4);

  m_pAccelStruct->ClearScene();
  m_pAccelStruct->AddInstance(geomId, LiteMath::float4x4());
  m_pAccelStruct->CommitScene();
}

void TestClass::kernel_InitEyeRay(uint* flags, float4* rayPosAndNear, float4* rayDirAndFar, uint tidX, uint tidY) 
{
  const float x = float(tidX)*m_widthInv;
  const float y = float(tidY)*m_heightInv;
  *(rayPosAndNear) = float4(x, y, -1.0f, 0.0f);
  *(rayDirAndFar ) = float4(0, 0, 1, MAXFLOAT);
  *flags           = 0;
}

void TestClass::kernel_RayTrace(const float4* rayPosAndNear, float4* rayDirAndFar, 
                                int* out_hit, uint tidX, uint tidY)
{
  CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(*rayPosAndNear, *rayDirAndFar);
  *out_hit = hit.primId;
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

uint32_t BFRayTrace::AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, 
                                         const uint32_t* a_triIndices, size_t a_indNumber, 
                                         BuildQuality a_qualityLevel, size_t vByteStride)
{
  const float4* verts2 = (const float4*)a_vpos3f;
  trivets = std::vector<float4>(verts2, verts2+a_vertNumber);
  return 0;
} 

CRT_Hit BFRayTrace::RayQuery_NearestHit(float4 rayPosAndNear, float4 rayDirAndFar)
{
  const float3 rayPos  = to_float3(rayPosAndNear);
  const float3 ray_dir = to_float3(rayDirAndFar);

  const float tNear = rayPosAndNear.w;
  const float tFar  = rayDirAndFar.w + testOffset;
  
  int hitId = -1;
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
  CRT_Hit res;
  res.primId = hitId;
  return res;
}
