#ifndef ISPC_MATH_H
#define ISPC_MATH_H

typedef float<2> float2;
typedef float<3> float3;
typedef float<4> float4;

typedef int<2>   int2;
typedef int<3>   int3;
typedef int<4>   int4;

typedef uint<2>  uint2;
typedef uint<3>  uint3;
typedef uint<4>  uint4;

typedef uint8    uint8_t;
typedef uint16   uint16_t;
typedef uint32   uint32_t;

typedef int8     int8_t;
typedef int16    int16_t;
typedef int32    int32_t;

#define __global 

#ifndef MAXFLOAT
#define MAXFLOAT 1e37f
#endif

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

//static inline float3 reflect(const float3 dir, const float3 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }

static inline float  fract(float x) { return x - floor(x); }
//static inline float  clamp(float u, float a, float b) { return min(max(a, u), b); }
static inline float4 clamp(const float4 u, float a, float b) { return make_float4(clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b), clamp(u.w, a, b)); }

typedef struct float3x3T
{
  float3 row[3];
} float3x3;

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

#define constexpr const static

static inline float dot (float4 a, float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
static inline float dot3(float4 a, float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float dot4(float4 a, float4 b)  { return dot(a,b); } 
static inline float dot (float3 a, float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }

static inline float  length(float3 a) { return sqrt(dot(a,a)); }
static inline float3 normalize(float3 a) { float lenInv = 1.0f/length(a); return a*lenInv; }

static inline float3 shuffle_xzy(float3 a) { return make_float3(a.x, a.z, a.y); }
static inline float3 shuffle_yxz(float3 a) { return make_float3(a.y, a.x, a.z); }
static inline float3 shuffle_yzx(float3 a) { return make_float3(a.y, a.z, a.x); }
static inline float3 shuffle_zxy(float3 a) { return make_float3(a.z, a.x, a.y); }
static inline float3 shuffle_zyx(float3 a) { return make_float3(a.z, a.y, a.x); }
static inline float3 cross(float3 a, float3 b) 
{
  const float3 a_yzx = shuffle_yzx(a);
  const float3 b_yzx = shuffle_yzx(b);
  return shuffle_yzx(a*b_yzx - a_yzx*b);
}


static inline float atomic_max_global(uniform float * uniform ptr, uniform float value) 
{ 
  uniform float cur;
  while (value > (cur = *ptr))
    value = atomic_compare_exchange_global(ptr, cur, value);
  return value;
}

static inline float atomic_min_global(uniform float * uniform ptr, uniform float value) 
{ 
  uniform float cur;
  while (value < (cur = *ptr))
    value = atomic_compare_exchange_global(ptr, cur, value);
  return value;
}

static inline float as_float(int x)    { return floatbits(x); }
static inline float as_float(uint32 x) { return floatbits(x); }

static inline uint32 as_uint32(float x) { return intbits(x); }


static inline uniform float3 reduce_add(float3 v)
{
  uniform float3 res;
  res.x = reduce_add(v.x);
  res.y = reduce_add(v.y);
  res.z = reduce_add(v.z);
  return res;
}

#endif
