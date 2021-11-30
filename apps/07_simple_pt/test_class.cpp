#include "test_class.h"
#include "include/crandom.h"

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

  const float3 rayDir = EyeRayDir(x, y, WIN_WIDTH, WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = m_camPos;
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}

void TestClass::kernel_InitEyeRay2(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar,
                                                                   float4* accumColor,    float4* accumuThoroughput) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,0);

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  const float3 rayDir = EyeRayDir(x, y, WIN_WIDTH, WIN_HEIGHT, m_worldViewProjInv); 
  const float3 rayPos = m_camPos;
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}


bool TestClass::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                Lite_Hit* out_hit, float2* out_bars)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
  
  Lite_Hit res;
  res.primId = hit.primId;
  res.instId = hit.instId;
  res.geomId = hit.geomId;
  res.t      = hit.t;

  float2 baricentrics = float2(hit.coords[0], hit.coords[1]);

  // intersect flat light under roof
  {
    const float tLightHit  = (m_lightGeom.boxMax.y - rayPos.y)/std::max(rayDir.y, 1e-6f);
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
 
  *out_hit  = res;
  *out_bars = baricentrics;
  return (res.primId != -1) && (res.t < rayDir.w);
  
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  out_pakedXY[pitchOffset(tidX,tidY)] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(*a_accumColor);
}

void TestClass::kernel_GetRayColor(uint tid, const Lite_Hit* in_hit, uint* out_color)
{
  if(in_hit->geomId == HIT_FLAT_LIGHT_GEOM)
  {
    out_color[tid] = RealColorToUint32_f3(float3(1,1,1));
  }
  else
  {
    const uint32_t mtId = m_materialIds[in_hit->primId];
    const float4 mdata  = m_materials[mtId];
    out_color[tid]      = RealColorToUint32_f3(to_float3(mdata)); 
  }
}

void TestClass::kernel_NextBounce(uint tid, const Lite_Hit* in_hit, const float2* in_bars, 
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput)
{
  // process light hit case
  //
  if(in_hit->geomId == HIT_FLAT_LIGHT_GEOM)
  {
    const float lightIntensity = m_materials[m_emissiveMaterialId].w;
    if(dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f)
      *accumColor = (*accumThoroughput)*lightIntensity;
    else
      *accumColor = float4(0,0,0,0);
    return;
  }
  
  // process surcase hit case
  //
  const Lite_Hit lHit  = *in_hit;
  const float3 ray_dir = to_float3(*rayDirAndFar);
  
  SurfaceHit hit;
  hit.pos  = to_float3(*rayPosAndNear) + lHit.t*ray_dir;
  hit.norm = EvalSurfaceNormal(ray_dir, lHit.primId, *in_bars, m_indicesReordered.data(), m_vNorm4f.data());

  RandomGen gen     = m_randomGens[tid];
  const float2 uv   = rndFloat2_Pseudo(&gen);
  m_randomGens[tid] = gen;

  const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hit.norm, hit.norm, 1.0f);
  const float  cosTheta = dot(newDir, hit.norm);

  const uint32_t mtId = m_materialIds[in_hit->primId];
  const float4 mdata  = m_materials[mtId];

  const float  pdfVal  = cosTheta * INV_PI;
  const float3 brdfVal = (cosTheta > 1e-5f) ? to_float3(mdata) * INV_PI : float3(0,0,0);
  const float3 bxdfVal = brdfVal * (1.0f / std::max(pdfVal, 1e-10f));
  
  *rayPosAndNear    = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
  *rayDirAndFar     = to_float4(newDir, MAXFLOAT);
  *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f);
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
  float2   baricentrics; 
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return;
  
  kernel_GetRayColor(tid, &hit, out_color);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*WIN_WIDTH+x] += *a_accumColor;
}

void TestClass::NaivePathTrace(uint tid, uint a_maxDepth, uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput);

  Lite_Hit hit; 
  float2   baricentrics; 

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
      break;
    
    kernel_NextBounce(tid, &hit, &baricentrics, 
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

  //test.LoadScene("lucy.bvh", "lucy.vsgf");
  test.LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.bvh", "../10_virtual_func_rt_test1/cornell_collapsed.vsgf", false);

  // test simple ray casting
  //
  //#pragma omp parallel for default(shared)
  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    test.CastSingleRay(i, packedXY.data(), pixelData.data());

  SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  
  // now test path tracing
  //
  const int PASS_NUMBER           = 100;
  const int ITERS_PER_PASS_NUMBER = 4;
  for(int passId = 0; passId < PASS_NUMBER; passId++)
  {
    #pragma omp parallel for default(shared)
    for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    {
      for(int j=0;j<ITERS_PER_PASS_NUMBER;j++)
        test.NaivePathTrace(i, 6, packedXY.data(), realColor.data());
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
  
}
