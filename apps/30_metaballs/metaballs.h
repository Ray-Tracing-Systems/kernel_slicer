#pragma once
#include <vector>
#include <cstdint>
#include "LiteMath.h"

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uint;

// Ray tracing of metaballs (implicit surface of summed compact-support kernels) with reflections,
// hard shadows and a reflective checkerboard floor. IPV organization: every pass is a 2D/1D
// "parallel for" over pixels, the same way image filters are written.
//
//   Render(w, h, spp, out):
//     kernel2D_Trace      - per pixel: spp x spp supersampling, up to MAX_BOUNCES reflections -> m_hdr
//     kernel1D_AvgLogLum  - sum-reduction of log luminance (for auto exposure)          -> m_sumLogLum
//     kernel2D_ToneMap    - exposure from the reduction, Reinhard, gamma                 -> out (RGBA8)
class Metaballs
{
public:
  static constexpr int   MAX_BALLS    = 16;
  static constexpr int   MAX_BOUNCES  = 4;    // primary hit + 3 reflections
  static constexpr int   MARCH_STEPS  = 96;   // uniform steps inside the ray's influence interval
  static constexpr int   REFINE_STEPS = 8;    // bisection steps at a sign change
  
  Metaballs();

  void SetMaxImageSize(int w, int h);                      // host only, before CommitDeviceData()
  void SetBalls(const std::vector<float4>& a_posRadius,    // host only: xyz = center, w = influence radius
                const std::vector<float4>& a_colors);      //            rgb = albedo,  w = reflectivity

  virtual void Render(int w, int h, int a_spp, uint32_t* a_out [[size("w*h")]]);

  virtual void CommitDeviceData() {}                                                          // overridden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; } // overridden in generated class

  float GetSumLogLum() const { return m_sumLogLum; }       // result of the reduction, read back from GPU

protected:
  virtual void kernel2D_Trace    (int w, int h, int a_spp, float4* a_hdr);
  virtual void kernel1D_AvgLogLum(int a_size, const float4* a_hdr);
  virtual void kernel2D_ToneMap  (int w, int h, float a_invPixels, const float4* a_hdr, uint32_t* a_out);

  // helpers, called from kernels (translated to shader code)
  float  Field      (float3 p) const;                       // sum of kernels, surface is Field(p) == m_threshold
  float3 FieldNormal(float3 p) const;                       // -grad(Field), normalized
  float4 SurfaceColor(float3 p) const;                      // albedo/reflectivity blended by kernel weights
  float2 InfluenceInterval(float3 ro, float3 rd) const;     // [tmin, tmax] where the ray meets any influence sphere
  float  TraceBalls (float3 ro, float3 rd, float tMax) const; // hit distance or -1
  float3 TracePath  (float3 ro, float3 rd) const;           // radiance with reflections (iterative)
  float3 Sky        (float3 rd) const;

  std::vector<float4> m_ballPosRadius;   // xyz center, w influence radius
  std::vector<float4> m_ballColor;       // rgb albedo, w reflectivity
  std::vector<float4> m_hdr;             // intermediate HDR image
  int    m_ballsNum  = 0;
  float  m_threshold = 0.25f;

  float4 m_camPos;                       // camera basis, float4 to keep std140/std430 layout simple
  float4 m_camRight;
  float4 m_camUp;
  float4 m_camForward;
  float4 m_lightDir;                     // direction TO the light
  float  m_floorY    = -1.1f;

  float  m_sumLogLum = 0.0f;             // sum-reduction result
  float  m_time      = 0.0f;
};
