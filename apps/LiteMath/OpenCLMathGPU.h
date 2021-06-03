#ifndef OPENCL_MATH_GPU_H
#define OPENCL_MATH_GPU_H

#ifndef MAXFLOAT
#define MAXFLOAT 1e37f
#endif

typedef uint uint32_t;
typedef int  int32_t;

typedef struct float4x4T
{
  float4 m_col[4];
} float4x4;

static inline float4 make_float4(float a, float b, float c, float d)
{
  float4 res;
  res.x = a;
  res.y = b;
  res.z = c;
  res.w = d;
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

#define constexpr __constant static

#endif
