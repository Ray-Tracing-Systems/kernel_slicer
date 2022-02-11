#ifndef RTC_MATERIAL
#define RTC_MATERIAL

#include "BasicLogic.h"

enum BRDF_TYPES { BRDF_TYPE_LAMBERT         = 1, 
                  BRDF_TYPE_GGX             = 2, 
                  BRDF_TYPE_LAMBERT_GGX_MIX = 3,
                  BRDF_TYPE_LAMBERT_LIGHT_SOURCE = 4, };

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
  const float GP           = 2.0f / (1.0f + sqrt(1.0f + alpha * alpha * tan2));
  return GP;
}

static inline float3 GgxVndf(float3 wo, float roughness, float u1, float u2)
{
  // -- Stretch the view vector so we are sampling as though
  // -- roughness==1
  const float3 v = normalize(float3(wo.x * roughness, wo.y * roughness, wo.z));

  // -- Build an orthonormal basis with v, t1, and t2
  float3 t1,t2;
  CoordinateSystem(v, &t1, &t2);

  // -- Choose a point on a disk with each half of the disk weighted
  // -- proportionally to its projection onto direction v
  const float a = 1.0f / (1.0f + v.z);
  const float r = std::sqrt(u1);
  const float phi = (u2 < a) ? (u2 / a) * M_PI : M_PI + (u2 - a) / (1.0f - a) * M_PI;
  const float p1 = r * cos(phi);
  const float p2 = r * sin(phi) * ((u2 < a) ? 1.0f : v.z);

  // -- Calculate the normal in this stretched tangent space
  const float3 n = p1 * t1 + p2 * t2 + std::sqrt(std::max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

  // -- unstretch and normalize the normal
  return normalize(float3(roughness * n.x, roughness * n.y, std::max(0.0f, n.z)));
}

static inline float SmithGGXMasking(const float dotNV, float roughSqr)
{
  const float denomC = sqrt(roughSqr + (1.0f - roughSqr) * dotNV * dotNV) + dotNV;
  return 2.0f * dotNV / std::max(denomC, 1e-6f);
}

static inline float SmithGGXMaskingShadowing(const float dotNL, const float dotNV, float roughSqr)
{
  const float denomA = dotNV * std::sqrt(roughSqr + (1.0f - roughSqr) * dotNL * dotNL);
  const float denomB = dotNL * std::sqrt(roughSqr + (1.0f - roughSqr) * dotNV * dotNV);
  return 2.0f * dotNL * dotNV / std::max(denomA + denomB, 1e-6f);
}

#endif