#include "test_class.h"
#include "include/crandom.h"


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestClass::InitSpheresScene(int a_numSpheres, int a_seed)
{ 
  spheresPosRadius.resize(8);
  spheresMaterials.resize(8);

  spheresPosRadius[0] = float4(0,-10000.0f,0,9999.0f);
  spheresMaterials[0].color = float4(0.5,0.5,0.5, 0.0f);

  spheresPosRadius[1] = float4(0,0,-4,1);
  spheresMaterials[1].color = float4(1,1,1,5);

  const float col = 0.75f;
  const float eps = 0.00f;

  spheresPosRadius[2] = float4(-2,0,-4,1);
  spheresMaterials[2].color = float4(col,eps,eps,0);

  spheresPosRadius[3] = float4(+2,0,-4,1);
  spheresMaterials[3].color = float4(eps,col,col,0);

  spheresPosRadius[4] = float4(-1,1.5,-4.5,1);
  spheresMaterials[4].color = float4(col,col,eps,0);

  spheresPosRadius[5] = float4(+1,1.5,-4.5,1);
  spheresMaterials[5].color = float4(eps,eps,col,0);

  spheresPosRadius[6] = float4(-1,-0.5,-3,0.5);
  spheresMaterials[6].color = float4(eps,col,eps,0);

  spheresPosRadius[7] = float4(+1,-0.5,-3,0.5);
  spheresMaterials[7].color = float4(eps,col,eps,0);
}

int TestClass::LoadScene(const char* bvhPath, const char* meshPath)
{
  std::fstream input_file;
  input_file.open(bvhPath, std::ios::binary | std::ios::in);
  if (!input_file.is_open())
  {
    std::cout << "BVH file error <" << bvhPath << ">\n";
    return 1;
  }
  struct BVHDataHeader
  {
    uint64_t node_length;
    uint64_t indices_length;
    uint64_t depth_length;
    uint64_t geom_id;
  };
  BVHDataHeader header;
  input_file.read((char *) &header, sizeof(BVHDataHeader));

//  m_bvhTree.geomID = header.geom_id;
  m_nodes.resize(header.node_length);
  m_intervals.resize(header.node_length);
  m_indicesReordered.resize(header.indices_length);
//  m_depthRanges.resize(header.depth_length);

  input_file.read((char *) m_nodes.data(), sizeof(BVHNode) * header.node_length);
  input_file.read((char *) m_intervals.data(), sizeof(Interval) * header.node_length);
  input_file.read((char *) m_indicesReordered.data(), sizeof(uint) * header.indices_length);
//  input_file.read((char *) m_bvhTree.depthRanges.data(), sizeof(Interval) * header.depth_length);

  std::fstream input_file_mesh;
  input_file_mesh.open(meshPath, std::ios::binary | std::ios::in);
  if (!input_file_mesh.is_open())
  {
    std::cout << "Mesh file error <" << meshPath << ">\n";
    return 1;
  }

  struct VSGFHeader
  {
    uint64_t fileSizeInBytes;
    uint32_t verticesNum;
    uint32_t indicesNum;
    uint32_t materialsNum;
    uint32_t flags;
  };
  VSGFHeader meshHeader;

  input_file_mesh.read((char*)&meshHeader, sizeof(VSGFHeader));
  m_mesh = SimpleMesh(meshHeader.verticesNum, meshHeader.indicesNum);

  input_file_mesh.read((char*)m_mesh.vPos4f.data(),  m_mesh.vPos4f.size()*sizeof(float)*4);

  if(!(meshHeader.flags & HAS_NO_NORMALS))
    input_file_mesh.read((char*)m_mesh.vNorm4f.data(), m_mesh.vNorm4f.size()*sizeof(float)*4);
  else
    memset(m_mesh.vNorm4f.data(), 0, m_mesh.vNorm4f.size()*sizeof(float)*4);  

  if(meshHeader.flags & HAS_TANGENT)
    input_file_mesh.read((char*)m_mesh.vTang4f.data(), m_mesh.vTang4f.size()*sizeof(float)*4);
  else
    memset(m_mesh.vTang4f.data(), 0, m_mesh.vTang4f.size()*sizeof(float)*4);

  input_file_mesh.read((char*)m_mesh.vTexCoord2f.data(), m_mesh.vTexCoord2f.size()*sizeof(float)*2);
  input_file_mesh.read((char*)m_mesh.indices.data(),    m_mesh.indices.size()*sizeof(unsigned int));
  input_file_mesh.read((char*)m_mesh.matIndices.data(), m_mesh.matIndices.size()*sizeof(unsigned int));
  input_file_mesh.close();

  m_vPos4f.resize(m_mesh.vPos4f.size());
  m_vPos4f = m_mesh.vPos4f;

  m_vNorm4f.resize(m_mesh.vNorm4f.size());
  m_vNorm4f = m_mesh.vNorm4f;

  return 0;

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

  const float3 rayDir = EyeRayDir((float)x, (float)y, (float)WIN_WIDTH, (float)WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = camPos;
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}

static bool RayBoxIntersection(float4 ray_pos, float4 ray_dir, const float boxMin[3], const float boxMax[3])
{
  float tmin = ray_pos.w;
  float tmax = ray_dir.w;

  ray_dir.x = 1.0f/ray_dir.x;
  ray_dir.y = 1.0f/ray_dir.y;
  ray_dir.z = 1.0f/ray_dir.z;

  float lo = ray_dir.x*(boxMin[0] - ray_pos.x);
  float hi = ray_dir.x*(boxMax[0] - ray_pos.x);

  tmin = fmin(lo, hi);
  tmax = fmax(lo, hi);

  float lo1 = ray_dir.y*(boxMin[1] - ray_pos.y);
  float hi1 = ray_dir.y*(boxMax[1] - ray_pos.y);

  tmin = fmax(tmin, fmin(lo1, hi1));
  tmax = fmin(tmax, fmax(lo1, hi1));

  float lo2 = ray_dir.z*(boxMin[2] - ray_pos.z);
  float hi2 = ray_dir.z*(boxMax[2] - ray_pos.z);

  tmin = fmax(tmin, fmin(lo2, hi2));
  tmax = fmin(tmax, fmax(lo2, hi2));

  return (tmin <= tmax) && (tmax > 0.f);
}

static void IntersectAllPrimitivesInLeaf(const float4 rayPosAndNear, const float4 rayDirAndFar,
                                         __global const uint* a_indices, uint a_start, uint a_count, __global const float4* a_vert,
                                         Lite_Hit* pHit)
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
    const float4 pvec  = cross(rayDirAndFar, edge2);
    const float4 tvec  = rayPosAndNear - A_pos;
    const float4 qvec  = cross(tvec, edge1);
    const float dotTmp = dot(to_float3(edge1), to_float3(pvec));
    const float invDet = 1.0f / (dotTmp > 1e-6f ? dotTmp : 1e-6f);

    const float v = dot(to_float3(tvec), to_float3(pvec))*invDet;
    const float u = dot(to_float3(qvec), to_float3(rayDirAndFar))*invDet;
    const float t = dot(to_float3(edge2), to_float3(qvec))*invDet;

    if (v > -1e-6f && u > -1e-6f && (u + v < 1.0f + 1e-6f) && t > rayPosAndNear.w && t < pHit->t)
    {
      pHit->t      = t;
      pHit->primId = triAddress/3;
    }
  }

}

bool TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                Lite_Hit* out_hit, const uint* indicesReordered, const float4* meshVerts)
{
  float4 rayPos = *rayPosAndNear;
  float4 rayDir = *rayDirAndFar ;

  Lite_Hit res;
  res.primId = -1;
  res.instId = -1;
  res.geomId = -1;
  res.t      = rayDir.w;

  uint nodeIdx = 0;
  while(nodeIdx < 0xFFFFFFFE)
  {
    const struct BVHNode currNode = m_nodes[nodeIdx];
    const bool intersects         = RayBoxIntersection(rayPos, rayDir, currNode.boxMin, currNode.boxMax);

    if(intersects && currNode.leftOffset == 0xFFFFFFFF) //leaf
    {
      struct Interval startCount = m_intervals[nodeIdx];
      IntersectAllPrimitivesInLeaf(rayPos, rayDir, indicesReordered, startCount.start*3, startCount.count*3, meshVerts, 
                                   &res);
    }

    nodeIdx = (currNode.leftOffset == 0xFFFFFFFF || !intersects) ? currNode.escapeIndex : currNode.leftOffset;
    nodeIdx = (nodeIdx == 0) ? 0xFFFFFFFE : nodeIdx;
  }
  
  *out_hit = res;
  return (res.primId != -1);
}

void TestClass::kernel_GetMaterialColor(uint tid, const Lite_Hit* in_hit, 
                                        uint* out_color)
{
  if(in_hit->primId != -1)
  {
    out_color[tid] = RealColorToUint32_f3(to_float3(spheresMaterials[in_hit->primId % 3].color));
  }
  else
    out_color[tid] = 0x00700000;
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_InitAccumData(uint tid, float4* accumColor, float4* accumuThoroughput)
{
  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,0);
}

void TestClass::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(*a_accumColor);
}

void TestClass::kernel_NextBounce(uint tid, const Lite_Hit* in_hit, 
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput)
{
  const Lite_Hit hit  = *in_hit;
  const float3 rayPos = to_float3(*rayPosAndNear);
  const float3 rayDir = to_float3(*rayDirAndFar );

  if( IsMtlEmissive(&spheresMaterials[hit.primId % 3]) )
  {
    float4 emissiveColor = to_float4(GetMtlEmissiveColor(&spheresMaterials[hit.primId % 3]), 0.0f);
    *accumColor = emissiveColor*(*accumThoroughput);
    return;
  }

  //const float3 sphPos    = to_float3(spheresPosRadius[hit.primId % 3]);
  const float3 diffColor = GetMtlDiffuseColor(&spheresMaterials[hit.primId % 3]);

  const float3 hitPos  = rayPos + rayDir*hit.t;
  //const float3 hitNorm = normalize(hitPos - sphPos);
  const float3 hitNorm = make_float3(m_vNorm4f[hit.primId].x, m_vNorm4f[hit.primId].y, m_vNorm4f[hit.primId].z);

  RandomGen gen = m_randomGens[tid];
  const float2 uv = rndFloat2_Pseudo(&gen);
  m_randomGens[tid] = gen;

  const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hitNorm, hitNorm, 1.0f);
  const float  cosTheta = dot(newDir, hitNorm);

  const float pdfVal   = cosTheta * INV_PI;
  const float3 brdfVal = (cosTheta > 1e-5f) ? diffColor * INV_PI : make_float3(0,0,0);
  const float3 bxdfVal = brdfVal * (1.0f / fmax(pdfVal, 1e-10f));

  const float3 newPos = OffsRayPos(hitPos, hitNorm, newDir);  

  *rayPosAndNear    = to_float4(newPos, 0.0f);
  *rayDirAndFar     = to_float4(newDir, MAXFLOAT);
  *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*WIN_WIDTH+x] += *a_accumColor;
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
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit;
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar,
                      &hit, m_indicesReordered.data(), m_vPos4f.data()))
    return;
  
  kernel_GetMaterialColor(tid, &hit, out_color);
}

void TestClass::StupidPathTrace(uint tid, uint a_maxDepth, uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  kernel_InitAccumData(tid, &accumColor, &accumThoroughput);

  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, m_indicesReordered.data(), m_vPos4f.data()))
      break;

    kernel_NextBounce(tid, &hit, 
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput);
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
  TestClass test(WIN_WIDTH*WIN_HEIGHT);

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

  test.LoadScene("lucy.bvh", "lucy.vsgf");
  // test simple ray casting
  //
  #pragma omp parallel for default(shared)
  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    test.CastSingleRay(i, packedXY.data(), pixelData.data());

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  //return;

  /*
  // now test path tracing
  //
  const int PASS_NUMBER = 100;
  const int ITERS_PER_PASS_NUMBER = 4;
  for(int passId = 0; passId < PASS_NUMBER; passId++)
  {
    #pragma omp parallel for default(shared)
    for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    {
      for(int j=0;j<ITERS_PER_PASS_NUMBER;j++)
        test.StupidPathTrace(i, 6, packedXY.data(), realColor.data());
    }

    if(passId%10 == 0)
    {
      const float progress = 100.0f*float(passId)/float(PASS_NUMBER);
      std::cout << "progress = " << progress << "%   \r";
      std::cout.flush();
    }
  }
  
  //std::cout << std::endl;

  const float normConst = 1.0f/float(PASS_NUMBER*ITERS_PER_PASS_NUMBER);
  const float invGamma  = 1.0f / 2.2f;

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    float4 color = realColor[i]*normConst;
    color.x      = powf(color.x, invGamma);
    color.y      = powf(color.y, invGamma);
    color.z      = powf(color.z, invGamma);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }
  SaveBMP("zout_cpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  */

  return;
}