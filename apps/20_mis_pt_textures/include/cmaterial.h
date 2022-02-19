#ifndef RTC_MATERIAL
#define RTC_MATERIAL

#include "cglobals.h"

enum BRDF_TYPES { BRDF_TYPE_LAMBERT         = 1, 
                  BRDF_TYPE_GGX             = 2, 
                  BRDF_TYPE_DIELECTRIC      = 3,
                  BRDF_TYPE_METALL          = 4,
                  BRDF_TYPE_GLTF            = 5,
                  BRDF_TYPE_GLASS           = 6,
                  BRDF_TYPE_MIRROR          = 7,

                  BRDF_TYPE_LAMBERT_LIGHT_SOURCE = 0xEFFFFFFF };

struct PlainMaterial
{
  uint  brdfType;      ///<
  float intensity;     ///< intensity for lights, take coloe from diffuse
  float alpha;         ///< blend factor between lambert and reflection : alpha*diffuse + (1.0f-alpha)*diffuse
  float diffuse   [3]; ///< color for both lambert and emissive lights
  float reflection[3]; ///<
  float glosiness;     ///<
  uint  lightId;       ///< identifier of light if this material is light
};

static inline float3 SphericalDirectionPBRT(const float sintheta, const float costheta, const float phi) 
{ 
  return make_float3(sintheta * cos(phi), sintheta * sin(phi), costheta); 
}

static inline float GGX_Distribution(const float cosThetaNH, const float alpha)
{
  const float alpha2  = alpha * alpha;
  const float NH_sqr  = clamp(cosThetaNH * cosThetaNH, 0.0f, 1.0f);
  const float den     = NH_sqr * alpha2 + (1.0f - NH_sqr);
  return alpha2 / std::max((float)(M_PI) * den * den, 1e-6f);
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
  const float tan2         = (1.0f - cosTheta_sqr) / std::max(cosTheta_sqr, 1e-6f);
  const float GP           = 2.0f / (1.0f + std::sqrt(1.0f + alpha * alpha * tan2));
  return GP;
}

#endif