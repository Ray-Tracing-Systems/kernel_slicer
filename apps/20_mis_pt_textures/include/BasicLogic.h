#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#define LAYOUT_STD140
#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct Lite_HitT
{
  float t;
  int   primId; 
  int   instId;
  int   geomId;
} Lite_Hit;

typedef struct SurfaceHitT
{
  float3 pos;
  float3 norm;
}SurfaceHit;

struct RectLightSource
{
  float4 pos;
  float4 intensity;
  float4 norm;
  float2 size;
  float2 dummy;
};

static inline float3 EyeRayDir(float x, float y, float w, float h, float4x4 a_mViewProjInv) // g_mViewProjInv
{
  float4 pos = make_float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                           -2.0f * (y + 0.5f) / h + 1.0f, 
                            0.0f, 
                            1.0f );

  pos = mul4x4x4(a_mViewProjInv, pos);
  pos /= pos.w;

  pos.y *= (-1.0f);

  return normalize(to_float3(pos));
}


static inline uint RealColorToUint32_f3(float3 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  unsigned char red = (unsigned char)r, green = (unsigned char)g, blue = (unsigned char)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

static inline uint RealColorToUint32(float4 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  float  a = real_color.w*255.0f;

  unsigned char red   = (unsigned char)r;
  unsigned char green = (unsigned char)g;
  unsigned char blue  = (unsigned char)b;
  unsigned char alpha = (unsigned char)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline void CoordinateSystem(float3 v1, float3* v2, float3* v3)
{
  float invLen = 1.0f;

  if (fabs(v1.x) > fabs(v1.y))
  {
    invLen = 1.0f / sqrt(v1.x*v1.x + v1.z*v1.z);
    (*v2)  = make_float3((-1.0f) * v1.z * invLen, 0.0f, v1.x * invLen);
  }
  else
  {
    invLen = 1.0f / sqrt(v1.y * v1.y + v1.z * v1.z);
    (*v2)  = make_float3(0.0f, v1.z * invLen, (-1.0f) * v1.y * invLen);
  }

  (*v3) = cross(v1, (*v2));
}

//constexpr float M_PI     = 3.14159265358979323846f;
//constexpr float M_TWOPI  = 6.28318530717958647692f;
//constexpr float INV_PI   = 0.31830988618379067154f

constexpr float GEPSILON = 5e-6f ;
constexpr float DEPSILON = 1e-20f;

//enum THREAD_FLAGS { THREAD_IS_DEAD = 2147483648};

static inline float3 MapSampleToCosineDistribution(float r1, float r2, float3 direction, float3 hit_norm, float power)
{
  if(power >= 1e6f)
    return direction;

  const float sin_phi = sin(M_TWOPI * r1);
  const float cos_phi = cos(M_TWOPI * r1);

  //sincos(2.0f*r1*3.141592654f, &sin_phi, &cos_phi);

  const float cos_theta = pow(1.0f - r2, 1.0f / (power + 1.0f));
  const float sin_theta = sqrt(1.0f - cos_theta*cos_theta);

  float3 deviation;
  deviation.x = sin_theta*cos_phi;
  deviation.y = sin_theta*sin_phi;
  deviation.z = cos_theta;

  float3 ny = direction, nx, nz;
  CoordinateSystem(ny, &nx, &nz);

  {
    float3 temp = ny;
    ny = nz;
    nz = temp;
  }

  float3 res = nx*deviation.x + ny*deviation.y + nz*deviation.z;

  float invSign = dot(direction, hit_norm) > 0.0f ? 1.0f : -1.0f;

  if (invSign*dot(res, hit_norm) < 0.0f) // reflected ray is below surface #CHECK_THIS
  {
    res = (-1.0f)*nx*deviation.x + ny*deviation.y - nz*deviation.z;
    //belowSurface = true;
  }

  return res;
}

static inline float epsilonOfPos(float3 hitPos) { return fmax(fmax(fabs(hitPos.x), fmax(fabs(hitPos.y), fabs(hitPos.z))), 2.0f*GEPSILON)*GEPSILON; }

/**
\brief offset reflected ray position by epsilon;
\param  a_hitPos      - world space position on surface
\param  a_surfaceNorm - surface normal at a_hitPos
\param  a_sampleDir   - ray direction in which we are going to trace reflected ray
\return offseted ray position
*/
static inline float3 OffsRayPos(const float3 a_hitPos, const float3 a_surfaceNorm, const float3 a_sampleDir)
{
  const float signOfNormal2 = dot(a_sampleDir, a_surfaceNorm) < 0.0f ? -1.0f : 1.0f;
  const float offsetEps     = epsilonOfPos(a_hitPos);
  return a_hitPos + signOfNormal2*offsetEps*a_surfaceNorm;
}


static inline float3 SphericalDirectionPBRT(const float sintheta, const float costheta, const float phi) 
{ 
  return make_float3(sintheta * cos(phi), sintheta * sin(phi), costheta); 
}

static inline float GGX_Distribution(const float cosThetaNH, const float alpha)
{
  const float alpha2  = alpha * alpha;
  const float NH_sqr  = clamp(cosThetaNH * cosThetaNH, 0.0f, 1.0f);
  const float den     = NH_sqr * alpha2 + (1.0f - NH_sqr);
  return alpha2 / fmax((float)(M_PI) * den * den, 1e-6f);
}

static inline float GGX_GeomShadMask(const float cosThetaN, const float alpha)
{
  // Height - Correlated G.
  //const float tanNV      = sqrt(1.0f - cosThetaN * cosThetaN) / cosThetaN;
  //const float a          = 1.0f / (alpha * tanNV);
  //const float lambda     = (-1.0f + sqrt(1.0f + 1.0f / (a*a))) / 2.0f;
  //const float G          = 1.0f / (1.0f + lambda);

  // Optimized and equal to the commented-out formulas on top.
  const float cosTheta_sqr = clamp(cosThetaN*cosThetaN, 0.0f, 1.0f);
  const float tan2         = (1.0f - cosTheta_sqr) / fmax(cosTheta_sqr, 1e-6f);
  const float GP           = 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
  return GP;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float3 mul3x3(float4x4 m, float3 v)
{ 
  return to_float3(m*to_float4(v, 0.0f));
}

static inline float3 mul4x3(float4x4 m, float3 v)
{
  return to_float3(m*to_float4(v, 1.0f));
}

static inline void transform_ray3f(float4x4 a_mWorldViewInv, float3* ray_pos, float3* ray_dir) 
{
  float3 pos  = mul4x3(a_mWorldViewInv, (*ray_pos));
  float3 pos2 = mul4x3(a_mWorldViewInv, ((*ray_pos) + 100.0f*(*ray_dir)));

  float3 diff = pos2 - pos;

  (*ray_pos)  = pos;
  (*ray_dir)  = normalize(diff);
}

static inline float PdfAtoW(const float aPdfA, const float aDist, const float aCosThere)
{
  return (aPdfA*aDist*aDist) / std::max(aCosThere, 1e-30f);
}

static inline float maxcomp(float3 v) { return std::max(v.x, std::max(v.y, v.z)); }

static inline float misHeuristicPower1(float p) { return std::isfinite(p) ? std::abs(p) : 0.0f; }
static inline float misWeightHeuristic(float a, float b)
{
  const float w = misHeuristicPower1(a) / std::max(misHeuristicPower1(a) + misHeuristicPower1(b), 1e-30f);
  return std::isfinite(w) ? w : 0.0f;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
\brief This structure is used as transit to pass MIS-weights-important-data from previouce bounce to current (or from current to next).

*/
typedef struct MisDataT
{
  float matSamplePdf; ///< previous angle pdf (pdfW) that were used for sampling material. if < 0, then material sample was pure specular  
} MisData;

static inline bool isSpecular(const MisData* data) { return (data->matSamplePdf < 0.0f); }

static inline MisData makeInitialMisData()
{
  MisData data;
  data.matSamplePdf = 1.0f;
  return data;
}


#endif