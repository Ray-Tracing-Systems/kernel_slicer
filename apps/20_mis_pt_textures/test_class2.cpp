#include "test_class.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

float Integrator::LightPdfSelectRev(int a_lightId) 
{ 
  return 1.0f; 
}

float Integrator::LightEvalPDF(int a_lightId, float3 illuminationPoint, float3 ray_dir, const SurfaceHit* pSurfaceHit)
{
  const float3 lpos   = pSurfaceHit->pos;
  const float3 lnorm  = pSurfaceHit->norm;
  const float hitDist = length(illuminationPoint - lpos);
  const float pdfA    = 1.0f / (4.0f*m_light.size.x*m_light.size.y);
  const float cosVal  = std::max(dot(ray_dir, -1.0f*lnorm), 0.0f);
  return PdfAtoW(pdfA, hitDist, cosVal);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

BsdfSample Integrator::MaterialSampleAndEval(int a_materialId, float4 rands, float3 v, float3 n)
{
  const uint   type      = m_materials[a_materialId].brdfType;
  const float3 color     = float3(m_materials[a_materialId].baseColor[0], m_materials[a_materialId].baseColor[1], m_materials[a_materialId].baseColor[2]);
  const float  roughness = 1.0f - m_materials[a_materialId].glosiness;
  const float  alpha     = m_materials[a_materialId].alpha;

  // TODO: read color     from texture
  // TODO: read roughness from texture
  // TODO: read alpha     from texture

  BsdfSample res;
  switch(type)
  {
    case BRDF_TYPE_GLTF:
    {
      const float3 ggxDir = ggxSample(float2(rands.x, rands.y), v, n, roughness);
      const float  ggxPdf = ggxEvalPDF (ggxDir, v, n, roughness); 
      const float  ggxVal = ggxEvalBSDF(ggxDir, v, n, roughness);
      
      const float3 lambertDir = lambertSample(float2(rands.x, rands.y), v, n);
      const float  lambertPdf = lambertEvalPDF(lambertDir, v, n);
      const float  lambertVal = lambertEvalBSDF(lambertDir, v, n);

      const float3 h = 0.5f*(v - ggxDir); // half vector.
      
      // TODO: check if glosiness in 1 (roughness is 0), use spetial case mirror brdf

      // (1) select between metal and dielectric via rands.z
      //
      float pdfSelect = 1.0f;
      if(rands.z < alpha) // select metall
      {
        pdfSelect = alpha;
        const float3 F = gltfConductorFresnel(color, dot(h,v));
        res.direction = ggxDir;
        res.color     = ggxVal*F*(1.0f/std::max(pdfSelect, 1e-4f));
        res.pdf       = ggxPdf;
      }
      else                // select dielectric
      {
        pdfSelect = 1.0f - alpha;
        
        // (2) select between specular and diffise via rands.w
        //
        const float fDielectricInv = gltfFresnelMix2(dot(h,v));
        if(rands.w < fDielectricInv)           // lambert
        {
          pdfSelect *= fDielectricInv;
          res.direction = lambertDir;
          res.color     = lambertVal*(1.0f/std::max(pdfSelect, 1e-4f))*color;
          res.pdf       = lambertPdf;
        }
        else
        {
          pdfSelect *= (1.0f-fDielectricInv); // specular
          res.direction = ggxDir;
          res.color     = ggxVal*(1.0f/std::max(pdfSelect, 1e-4f))*color;
          res.pdf       = ggxPdf;
        }
      }
    }
    break;
    case BRDF_TYPE_GGX:
    { 
      res.direction = ggxSample(float2(rands.x, rands.y), v, n, roughness);
      res.color     = ggxEvalBSDF(res.direction, v, n, roughness)*color;
      res.pdf       = ggxEvalPDF(res.direction, v, n, roughness);
    }
    break;
    case BRDF_TYPE_MIRROR:
    {
      res.direction = reflect(v, n);
      // BSDF is multiplied (outside) by cosThetaOut.
      // For mirrors this shouldn't be done, so we pre-divide here instead.
      //
      const float cosThetaOut = dot(res.direction, n);
      res.color     = cosThetaOut*color;
      res.pdf       = 1.0f;
    }
    break;
    case BRDF_TYPE_LAMBERT:
    default:
    {
      res.direction = lambertSample(float2(rands.x, rands.y), v, n);
      res.color     = lambertEvalBSDF(res.direction, v, n)*color;
      res.pdf       = lambertEvalPDF(res.direction, v, n);
    }
    break;
  }

  return res;
}

BsdfEval Integrator::MaterialEval(int a_materialId, float3 l, float3 v, float3 n)
{
  const uint type       = m_materials[a_materialId].brdfType;
  const float3 color    = float3(m_materials[a_materialId].baseColor[0], m_materials[a_materialId].baseColor[1], m_materials[a_materialId].baseColor[2]);
  const float roughness = 1.0f - m_materials[a_materialId].glosiness;
  const float  alpha    = m_materials[a_materialId].alpha;

  // TODO: read color     from texture
  // TODO: read roughness from texture
  // TODO: read alpha     from texture

  BsdfEval res;
  switch(type)
  {
    case BRDF_TYPE_GLTF:
    {
      const float ggxVal = ggxEvalBSDF(l, v, n, roughness);
      const float ggxPdf = ggxEvalPDF (l, v, n, roughness);
      
      const float lambertVal = lambertEvalBSDF(l, v, n);
      const float lambertPdf = lambertEvalPDF (l, v, n);
      
      const float3 h = 0.5f*(v + l);

      const float3 specularColor   = color*ggxVal;     // (1) eval metal and (same) specular component
      const float3 diffuseColor    = color*lambertVal; // (2) eval diffise component
      //const float3 dielectricColor = gltfFresnelMix(diffuseColor, specularColor, 1.5f, dot(v,h));  // (3) eval dielectric component
      const float  fDielectricInv  = gltfFresnelMix2(dot(h,v));
      const float  dielectricPdf   = fDielectricInv*lambertPdf + (1.0f-fDielectricInv)*ggxPdf;
      const float  dielectricVal   = fDielectricInv*lambertVal + (1.0f-fDielectricInv)*ggxVal;

      res.color = alpha*specularColor + (1.0f - alpha)*dielectricVal*color; // (4) accumulate final color and pdf
      res.pdf   = alpha*ggxPdf        + (1.0f - alpha)*dielectricPdf;       // (4) accumulate final color and pdf
    }
    break;
    case BRDF_TYPE_GGX:
    {
      res.color = ggxEvalBSDF(l, v, n, roughness)*color;
      res.pdf   = ggxEvalPDF (l, v, n, roughness);
    }
    break;
    case BRDF_TYPE_MIRROR:
    {
      res.color = float3(0,0,0);
      res.pdf   = 0.0f;
    }
    break;
    case BRDF_TYPE_LAMBERT:
    default:
    {
      res.color = lambertEvalBSDF(l, v, n)*color;
      res.pdf   = lambertEvalPDF (l, v, n);
    }
    break;
  }
  return res;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  kernel_PackXY(tidX, tidY, out_pakedXY);
}

void Integrator::CastSingleRay(uint tid, const uint* in_pakedXY, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit; 
  float2   baricentrics; 
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return;
  
  kernel_GetRayColor(tid, &hit, in_pakedXY, out_color);
}

void Integrator::NaivePathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &rayFlags);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_ContributeToImage(tid, &accumColor, &gen, in_pakedXY, 
                           out_color);
}

void Integrator::PathTrace(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color)
{
  float4 accumColor, accumThoroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags;
  kernel_InitEyeRay2(tid, in_pakedXY, &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &rayFlags);

  for(int depth = 0; depth < a_maxDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags, 
                             &gen, &shadeColor);

    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThoroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_ContributeToImage(tid, &accumColor, &gen, in_pakedXY, 
                           out_color);
                           
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::PackXYBlock(uint tidX, uint tidY, uint* out_pakedXY, uint a_passNum)
{
  #pragma omp parallel for default(shared)
  for(int y=0;y<tidY;y++)
    for(int x=0;x<tidX;x++)
      PackXY(x, y, out_pakedXY);
}

void Integrator::CastSingleRayBlock(uint tid, const uint* in_pakedXY, uint* out_color, uint a_passNum)
{
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    CastSingleRay(i, in_pakedXY, out_color);
}

void Integrator::NaivePathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passNum;j++)
      NaivePathTrace(i, 6, in_pakedXY, out_color);
  naivePtTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void Integrator::PathTraceBlock(uint tid, uint a_maxDepth, const uint* in_pakedXY, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for default(shared)
  for(uint i=0;i<tid;i++)
    for(int j=0;j<a_passNum;j++)
      PathTrace(i, 6, in_pakedXY, out_color);
  shadowPtTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()/1000.f;
}

void Integrator::GetExecutionTime(const char* a_funcName, float a_out[4])
{
  if(std::string(a_funcName) == "NaivePathTrace" || std::string(a_funcName) == "NaivePathTraceBlock")
    a_out[0] = naivePtTime;
  else if(std::string(a_funcName) == "PathTrace" || std::string(a_funcName) == "PathTraceBlock")
    a_out[0] = shadowPtTime;
}
