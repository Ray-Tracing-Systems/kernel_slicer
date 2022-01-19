#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

void TestClass::InitRandomGens(int a_maxThreads)
{
  m_randomGens.resize(a_maxThreads);
  #pragma omp parallel for default(shared)
  for(int i=0;i<a_maxThreads;i++)
    m_randomGens[i] = RandomGenInit(i);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float lambertEvalPDF (const float3 l, const float3 n) { return std::abs(dot(l, n)) * INV_PI; }

static inline float areaDiffuseLightEvalPDF(const RectLightSource* pLight, float3 rayDir, float hitDist)
{
  const float pdfA   = 1.0f / (pLight->size.x*pLight->size.y);
  const float cosVal = std::max(dot(rayDir, -1.0f*to_float3(pLight->norm)), 0.0f);
  return PdfAtoW(pdfA, hitDist, cosVal);
}

static inline float lightPdfSelectRev(const RectLightSource* pLight) { return 1.0f; }


void TestClass::kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayDir = EyeRayDir(x, y, m_winWidth, m_winHeight, m_projInv); 
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, 
                  &rayPos, &rayDir);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}

void TestClass::kernel_InitEyeRay2(uint tid, const uint* packedXY, 
                                   float4* rayPosAndNear, float4* rayDirAndFar,
                                   float4* accumColor,    float4* accumuThoroughput,
                                   RandomGen* gen) // 
{
  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,0);
  *gen               = m_randomGens[tid];

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayDir = EyeRayDir(x, y, m_winWidth, m_winHeight, m_projInv); 
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, 
                  &rayPos, &rayDir);
  
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
 
  *out_hit  = res;
  *out_bars = baricentrics;
  return (res.primId != -1);
}

void TestClass::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  const uint inBlockIdX = tidX % 8; // 8x8 blocks
  const uint inBlockIdY = tidY % 8; // 8x8 blocks
 
  const uint localIndex = inBlockIdY*8 + inBlockIdX;
  const uint wBlocks    = m_winWidth/8;

  const uint blockX     = tidX/8;
  const uint blockY     = tidY/8;
  const uint offset     = (blockX + blockY*wBlocks)*8*8 + localIndex;

  out_pakedXY[offset] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void TestClass::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(*a_accumColor);
}

void TestClass::kernel_GetRayColor(uint tid, const Lite_Hit* in_hit, const uint* in_pakedXY, uint* out_color)
{ 
  const Lite_Hit lhit = *in_hit;
  if(lhit.geomId == -1)
  {
    out_color[tid] = 0;
    return;
  }

  const uint32_t matId = m_matIdByPrimId[m_matIdOffsets[lhit.geomId] + lhit.primId];
  const float4 mdata   = m_materials[matId];
  const float3 color   = mdata.w > 0.0f ? clamp(float3(mdata.w,mdata.w,mdata.w), 0.0f, 1.0f) : to_float3(mdata);

  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;

  out_color[y*m_winWidth+x] = RealColorToUint32_f3(color); 
}



void TestClass::kernel_SampleLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, const Lite_Hit* in_hit, const float2* in_bars, 
                                         RandomGen* a_gen, float4* out_shadeColor)
{
  const Lite_Hit lhit = *in_hit;
  if(lhit.geomId == -1)
    return;
  
  const float3 ray_dir = to_float3(*rayDirAndFar);

  SurfaceHit hit;
  hit.pos  = to_float3(*rayPosAndNear) + lhit.t*ray_dir;
  hit.norm = EvalSurfaceNormal(lhit.primId, lhit.geomId, *in_bars, m_matIdOffsets.data(), m_vertOffset.data(), m_triIndices.data(), m_vNorm4f.data());
  
  // transform surface point with matrix and flip normal if needed
  {
    hit.norm = normalize(mul3x3(m_normMatrices[lhit.instId], hit.norm));
    const float flipNorm = dot(ray_dir,  hit.norm) > 0.001f ? -1.0f : 1.0f;
    hit.norm = flipNorm*hit.norm;
  }

  RandomGen gen   = *a_gen;
  const float2 uv = rndFloat2_Pseudo(&gen);
  *a_gen          = gen;  
  
  const float2 sampleOff = 2.0f*(float2(-0.5,-0.5) + uv)*m_light.size;
  const float3 samplePos = to_float3(m_light.pos) + float3(sampleOff.x, -1e-5f, sampleOff.y);
  const float  hitDist   = sqrt(dot(hit.pos - samplePos, hit.pos - samplePos));

  const float3 shadowRayDir = normalize(samplePos - hit.pos); // explicitSam.direction;
  const float3 shadowRayPos = hit.pos + hit.norm*std::max(maxcomp(hit.pos), 1.0f)*5e-6f;

  const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  
  if(!inShadow && dot(shadowRayDir, to_float3(m_light.norm)) < 0.0f)
  {
    const float pdfA    = 1.0f / (m_light.size.x*m_light.size.y);
    const float cosVal  = std::max(dot(shadowRayDir, (-1.0f)*to_float3(m_light.norm)), 0.0f);
    const float pdfW    = PdfAtoW(pdfA, hitDist, cosVal);
    const float3 samCol = M_PI*to_float3(m_light.intensity)/std::max(pdfW, 1e-6f); //////////////////////// Apply Pdf here, or outside of here ???
    
    const uint32_t matId = m_matIdByPrimId[m_matIdOffsets[lhit.geomId] + lhit.primId];
    const float4 mdata   = m_materials[matId];

    const float3 brdfVal    = to_float3(mdata)*INV_PI;
    const float cosThetaOut = std::max(dot(shadowRayDir, hit.norm), 0.0f);

    if(cosVal <= 0.0f)
      *out_shadeColor = float4(0.0f, 0.0f, 0.0f, pdfW);
    else
      *out_shadeColor = to_float4(samCol*brdfVal*cosThetaOut, pdfW);
  }
  else
    *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
}

void TestClass::kernel_NextBounce(uint tid, uint bounce, const Lite_Hit* in_hit, const float2* in_bars, const float4* in_shadeColor, 
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* misPrev)
{
  const Lite_Hit lhit = *in_hit;
  if(lhit.geomId == -1)
    return;

  const uint32_t matId = m_matIdByPrimId[m_matIdOffsets[lhit.geomId] + lhit.primId];
  const float4 mdata   = m_materials[matId];

  // process surcase hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  SurfaceHit hit;
  hit.pos  = to_float3(*rayPosAndNear) + lhit.t*ray_dir;
  hit.norm = EvalSurfaceNormal(lhit.primId, lhit.geomId, *in_bars, m_matIdOffsets.data(), m_vertOffset.data(), m_triIndices.data(), m_vNorm4f.data());

  // process light hit case
  //
  if(mdata.w > 0.0f)
  {
    if(m_intergatorType == INTEGRATOR_STUPID_PT)
    {
      const float lightIntensity = mdata.w;
      if(dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f)
        *accumColor = (*accumThoroughput)*lightIntensity;
      else
        *accumColor = float4(0,0,0,0);
    }
    else if(m_intergatorType == INTEGRATOR_MIS_PT) // #TODO: implement MIS weights
    {
      float misWeight = 1.0f;
      if(bounce > 0)
      {
        //const int lightId   = ...
        //const float lgtPdf  = lightPdfSelectRev(lightId)*lightEvalPDF(lightId, ray_pos, ray_dir, &surfElem);
        //const float bsdfPdf = misPrev->matSamplePdf;
        //if (bsdfPdf < 0.0f) // specular bounce
          //misWeight = 1.0f;
      }
      //if(dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f)
      //  *accumColor = (*accumThoroughput)*lightIntensity*misWeight;
      //else
      //  *accumColor = float4(0,0,0,0);
    }

    return;
  }
  
  // transform surface point with matrix and flip normal if needed
  {
    hit.norm = normalize(mul3x3(m_normMatrices[lhit.instId], hit.norm));
    const float flipNorm = dot(ray_dir,  hit.norm) > 0.001f ? -1.0f : 1.0f;
    hit.norm = flipNorm*hit.norm;
  }
  
  RandomGen gen   = *a_gen;
  const float2 uv = rndFloat2_Pseudo(&gen);
  *a_gen          = gen;

  const float3 newDir   = MapSampleToCosineDistribution(uv.x, uv.y, hit.norm, hit.norm, 1.0f);
  const float  cosTheta = dot(newDir, hit.norm);

  const float  pdfVal  = cosTheta * INV_PI;
  const float3 brdfVal = (cosTheta > 1e-5f) ? to_float3(mdata) * INV_PI : float3(0,0,0);
  const float3 bxdfVal = brdfVal * (1.0f / std::max(pdfVal, 1e-10f));
  
  MisData nextBounceData;               // remember current pdfW for next bounce
  nextBounceData.matSamplePdf = pdfVal; //
  *misPrev = nextBounceData;            //

  if(m_intergatorType == INTEGRATOR_STUPID_PT)
  {
    *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f); 
  }
  else if(m_intergatorType == INTEGRATOR_SHADOW_PT)
  {
    const float4 currThoroughput = *accumThoroughput;
    const float4 shadeColor      = *in_shadeColor;

    *accumColor += currThoroughput*shadeColor;
    *accumThoroughput = currThoroughput*cosTheta*to_float4(bxdfVal, 0.0f); 
  }
  else if(m_intergatorType == INTEGRATOR_MIS_PT) // #TODO: implement MIS weights
  {
    const float4 currThoroughput = *accumThoroughput;
    const float4 shadeColor      = *in_shadeColor;
    const float lgtPdf           = shadeColor.w;
    //const float bsdfPdf = EvalPDF(materialId, &hit, &explicitSam);
    //float misWeight = misWeightHeuristic(lgtPdf, bsdfPdf); 
    //if (lgtPdf < 0.0f)
    //  misWeight = 1.0f;
    //*accumColor += currThoroughput*shadeColor*misWeight;
    //*accumThoroughput = currThoroughput*cosTheta*to_float4(bxdfVal, 0.0f); 
  }

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
  *rayDirAndFar  = to_float4(newDir, MAXFLOAT);
  
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
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return;
  
  kernel_GetRayColor(tid, &hit, in_pakedXY, out_color);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const RandomGen* gen, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*m_winWidth+x] += *a_accumColor;
  m_randomGens[tid] = *gen;
}

void TestClass::NaivePathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    float2   baricentrics; 
    float4   shadeColor;
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
      break;
    
    kernel_NextBounce(tid, depth, &hit, &baricentrics, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &mis);
  }

  kernel_ContributeToImage(tid, &accumColor, &gen, in_pakedXY, 
                           out_color);
}

void TestClass::PathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    Lite_Hit hit;
    float2   baricentrics; 
    if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
      break;
    
    float4 shadeColorAndPdf; // pack pdf to color.w to save space; if pdf < 0.0, then say we have point light source
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics, &gen, 
                             &shadeColorAndPdf);

    kernel_NextBounce(tid, depth, &hit, &baricentrics, &shadeColorAndPdf,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &mis);
  }

  kernel_ContributeToImage(tid, &accumColor, &gen, in_pakedXY, 
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
  naivePtTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void TestClass::PathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passNum;j++)
      PathTrace(i, 6, in_pakedXY, out_color);
  shadowPtTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void TestClass::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "NaivePathTrace" || std::string(a_funcName) == "NaivePathTraceBlock")
    a_out[0] = naivePtTime;
  else if(std::string(a_funcName) == "PathTrace" || std::string(a_funcName) == "PathTraceBlock")
    a_out[0] = shadowPtTime;
}
