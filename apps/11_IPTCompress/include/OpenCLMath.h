#ifndef OPENCLMATH_H
#define OPENCLMATH_H

// (1) Dealing with code/math that differs for CPU and GPU
//

#ifdef __OPENCL_VERSION__
  #include "OpenCLMathGPU.h" // include some additional math or code you need to be applied in OpenCL kernels
#else
  #ifdef USE_CIRCLE_CC
  #include "CircleToCLMath.h"
  #else
  #include "OpenCLMathCPU.h"   // implementation of _same_ functions on the CPU
  using namespace LiteMath;
  #define __global 
  #endif
#endif

// (2) put you general logoc math code that will be same for CPU and GPU
//
#ifndef M_PI
#define M_PI          3.14159265358979323846f
#endif

#ifndef M_TWOPI
#define M_TWOPI       6.28318530717958647692f
#endif

#ifndef INV_PI
#define INV_PI        0.31830988618379067154f
#endif

#ifdef  INV_TWOPI
#define INV_TWOPI     0.15915494309189533577f
#endif 

static inline float2 make_float2(float a, float b)
{
  float2 res;
  res.x = a;
  res.y = b;
  return res;
}

static inline float3 make_float3(float a, float b, float c)
{
  float3 res;
  res.x = a;
  res.y = b;
  res.z = c;
  return res;
}

static inline float4 make_float4(float a, float b, float c, float d)
{
  float4 res;
  res.x = a;
  res.y = b;
  res.z = c;
  res.w = d;
  return res;
}

static inline float2 to_float2(float4 f4)
{
  float2 res;
  res.x = f4.x;
  res.y = f4.y;
  return res;
}

static inline float3 to_float3(float4 f4)
{
  float3 res;
  res.x = f4.x;
  res.y = f4.y;
  res.z = f4.z;
  return res;
}

static inline float4 to_float4(float3 v, float w)
{
  float4 res;
  res.x = v.x;
  res.y = v.y;
  res.z = v.z;
  res.w = w;
  return res;
}

static inline float4 mul4x4x4(float4x4 m, float4 v)
{
  float4 res;
  res.x = v.x * m.m_col[0].x + v.y * m.m_col[1].x + v.z * m.m_col[2].x + v.w * m.m_col[3].x;
  res.y = v.x * m.m_col[0].y + v.y * m.m_col[1].y + v.z * m.m_col[2].y + v.w * m.m_col[3].y;
  res.z = v.x * m.m_col[0].z + v.y * m.m_col[1].z + v.z * m.m_col[2].z + v.w * m.m_col[3].z;
  res.w = v.x * m.m_col[0].w + v.y * m.m_col[1].w + v.z * m.m_col[2].w + v.w * m.m_col[3].w;
  return res;
}

#endif
