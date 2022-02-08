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
                                   RandomGen* gen, uint* rayFlags) // 
{
  *accumColor        = make_float4(0,0,0,0);
  *accumuThoroughput = make_float4(1,1,1,0);
  *gen               = m_randomGens[tid];
  *rayFlags          = 0;

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

void TestClass::kernel_RayTrace2(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar,
                                 float4* out_hit1, float4* out_hit2, uint32_t* out_matId, uint* rayFlags)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  const CRT_Hit hit   = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);

  if(hit.geomId != uint32_t(-1))
  {
    const float2 uv       = float2(hit.coords[0], hit.coords[1]);
    const float3 hitPos   = to_float3(rayPos) + hit.t*to_float3(rayDir);

    const uint triOffset  = m_matIdOffsets[hit.geomId];
    const uint vertOffset = m_vertOffset  [hit.geomId];
  
    const uint A = m_triIndices[(triOffset + hit.primId)*3 + 0];
    const uint B = m_triIndices[(triOffset + hit.primId)*3 + 1];
    const uint C = m_triIndices[(triOffset + hit.primId)*3 + 2];
  
    const float3 A_norm = to_float3(m_vNorm4f[A + vertOffset]);
    const float3 B_norm = to_float3(m_vNorm4f[B + vertOffset]);
    const float3 C_norm = to_float3(m_vNorm4f[C + vertOffset]);
      
    float3 hitNorm     = (1.0f - uv.x - uv.y)*A_norm + uv.y*B_norm + uv.x*C_norm;
    float2 hitTexCoord = float2(0,0); // #TODO: evaluate hitTexCoord
  
    // transform surface point with matrix and flip normal if needed
    //
    hitNorm = normalize(mul3x3(m_normMatrices[hit.instId], hitNorm));
    const float flipNorm = dot(to_float3(rayDir), hitNorm) > 0.001f ? -1.0f : 1.0f;
    hitNorm = flipNorm*hitNorm;
  
    *out_matId = m_matIdByPrimId[m_matIdOffsets[hit.geomId] + hit.primId];
    *out_hit1  = to_float4(hitPos, hitTexCoord.x); 
    *out_hit2  = to_float4(hitNorm, hitTexCoord.y); 
  }
  else
    *out_matId = uint32_t(-1);

  *rayFlags = (hit.geomId != -1) ? 0 : 1;
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



void TestClass::kernel_SampleLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, const float4* in_hitPart1, const float4* in_hitPart2, const uint* a_materialId, 
                                         RandomGen* a_gen, float4* out_shadeColor)
{
  const uint32_t matId = *a_materialId;
  if(matId == uint32_t(-1))
    return;
  
  const float3 ray_dir = to_float3(*rayDirAndFar);

  SurfaceHit hit;
  hit.pos  = to_float3(*in_hitPart1);
  hit.norm = to_float3(*in_hitPart2);

  const float2 uv = rndFloat2_Pseudo(a_gen);
  
  const float2 sampleOff = 2.0f*(float2(-0.5f,-0.5f) + uv)*m_light.size;
  const float3 samplePos = to_float3(m_light.pos) + float3(sampleOff.x, -1e-5f, sampleOff.y);
  const float  hitDist   = sqrt(dot(hit.pos - samplePos, hit.pos - samplePos));

  const float3 shadowRayDir = normalize(samplePos - hit.pos); // explicitSam.direction;
  const float3 shadowRayPos = hit.pos + hit.norm*std::max(maxcomp(hit.pos), 1.0f)*5e-6f;

  const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  
  if(!inShadow && dot(shadowRayDir, to_float3(m_light.norm)) < 0.0f)
  {
    const float lightPickProb = 1.0f;

    const float pdfA    = 1.0f / (4.0f*m_light.size.x*m_light.size.y);
    const float cosVal  = std::max(dot(shadowRayDir, (-1.0f)*to_float3(m_light.norm)), 0.0f);
    const float lgtPdfW = PdfAtoW(pdfA, hitDist, cosVal);
    const float3 samCol = to_float3(m_light.intensity)/std::max(lgtPdfW, 1e-6f); //////////////////////// Apply Pdf here, or outside of here ???
  
    const float4 mdata  = m_materials[matId];

    const float3 brdfVal    = to_float3(mdata)*INV_PI;
    const float cosThetaOut = std::max(dot(shadowRayDir, hit.norm), 0.0f);

    float misWeight = 1.0f;
    if(m_intergatorType == INTEGRATOR_MIS_PT)
    {
      const float matPdfW = MaterialEvalPDF(matId, shadowRayDir, (-1.0f)*ray_dir, hit.norm); 
      misWeight = misWeightHeuristic(lgtPdfW, matPdfW);
    }

    if(cosVal <= 0.0f)
      *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
    else
      *out_shadeColor = to_float4((1.0f/lightPickProb)*samCol*brdfVal*cosThetaOut*misWeight, 0.0f);
  }
  else
    *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
}

void TestClass::kernel_NextBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const uint* a_materialId, const float4* in_shadeColor, 
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* misPrev, uint* rayFlags)
{
  const uint32_t matId = *a_materialId;
  if(matId == uint32_t(-1))
    return;

  const float4 mdata   = m_materials[matId];

  // process surcase hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  const float3 ray_pos = to_float3(*rayPosAndNear);

  SurfaceHit hit;
  hit.pos  = to_float3(*in_hitPart1);
  hit.norm = to_float3(*in_hitPart2);

  // process light hit case
  //
  if(mdata.w > 0.0f)
  {
    const float lightIntensity = mdata.w;
    float misWeight = 1.0f;
    if(m_intergatorType == INTEGRATOR_MIS_PT) 
    {
      if(bounce > 0)
      {
        const int lightId   = 0; // #TODO: get light id from material info
        const float lgtPdf  = LightPdfSelectRev(lightId)*LightEvalPDF(lightId, ray_pos, ray_dir, &hit);
        const float bsdfPdf = misPrev->matSamplePdf;
        misWeight           = misWeightHeuristic(bsdfPdf, lgtPdf);
        if (bsdfPdf < 0.0f) // specular bounce
          misWeight = 1.0f;
      }
    }
    else if(m_intergatorType == INTEGRATOR_SHADOW_PT)
      misWeight = 0.0f;

    if(dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f)
      *accumColor = (*accumThoroughput)*lightIntensity*misWeight;
    else
      *accumColor = float4(0,0,0,0);
    
    *rayFlags = 1;
    return;
  }
  
  const float2 uv       = rndFloat2_Pseudo(a_gen);
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
  else if(m_intergatorType == INTEGRATOR_SHADOW_PT || m_intergatorType == INTEGRATOR_MIS_PT)
  {
    const float4 currThoroughput = *accumThoroughput;
    const float4 shadeColor      = *in_shadeColor;

    *accumColor += currThoroughput*shadeColor;
    *accumThoroughput = currThoroughput*cosTheta*to_float4(bxdfVal, 0.0f); 
  }

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, newDir), 0.0f);
  *rayDirAndFar  = to_float4(newDir, MAXFLOAT);
}

void TestClass::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const RandomGen* gen, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;
 
  out_color[y*m_winWidth+x] += *a_accumColor;
  m_randomGens[tid] = *gen;
}
