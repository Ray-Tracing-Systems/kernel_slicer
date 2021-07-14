#ifndef OPENCLMATH_H
#define OPENCLMATH_H

// (1) Dealing with code/math that differs for CPU and GPU
//

#ifdef __OPENCL_VERSION__
  #include "OpenCLMathGPU.h" // include some additional math or code you need to be applied in OpenCL kernels
#else
  #ifdef KERNEL_SLICER
  #include "OpenCLMathCPU.h" // pure implementation of _same_ functions on the CPU without vector extensions (you may use it for CPU also for the case)
  #else
  #include "OpenCLMathVEX.h" // pure implementation of _same_ functions on the CPU with    vector extensions
  #endif
  using namespace LiteMath;
  #define __global 
#endif

// (2) put you general logic math code that will be same for CPU and GPU
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

typedef struct float3x3T
{
  float3 row[3];
} float3x3;

static inline float2 make_float2(float a, float b)
{
  float2 res;
  res.x = a;
  res.y = b;
  return res;
}

static inline uint2 make_uint2(uint a, uint b)
{
  uint2 res;
  res.x = a;
  res.y = b;
  return res;
}

static inline int2 make_int2(int a, int b)
{
  int2 res;
  res.x = a;
  res.y = b;
  return res;
}

static inline float2 to_float2(float4 f4)
{
  float2 res;
  res.x = f4.x;
  res.y = f4.y;
  return res;
}

static inline float3x3 make_float3x3(float3 a, float3 b, float3 c)
{
  float3x3 m;
  m.row[0] = a;
  m.row[1] = b;
  m.row[2] = c;
  return m;
}

static inline float3x3 make_float3x3_by_columns(float3 a, float3 b, float3 c)
{
  float3x3 m;
  m.row[0].x = a.x;
  m.row[1].x = a.y;
  m.row[2].x = a.z;

  m.row[0].y = b.x;
  m.row[1].y = b.y;
  m.row[2].y = b.z;

  m.row[0].z = c.x;
  m.row[1].z = c.y;
  m.row[2].z = c.z;
  return m;
}


static inline float3 mul3x3x3(float3x3 m, const float3 v)
{
  float3 res;
  res.x = m.row[0].x*v.x + m.row[0].y*v.y + m.row[0].z*v.z;
  res.y = m.row[1].x*v.x + m.row[1].y*v.y + m.row[1].z*v.z;
  res.z = m.row[2].x*v.x + m.row[2].y*v.y + m.row[2].z*v.z;
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