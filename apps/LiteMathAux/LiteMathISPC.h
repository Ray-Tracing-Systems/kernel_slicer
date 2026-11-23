#ifndef ISPC_MATH_H
#define ISPC_MATH_H

typedef float16  half;   // LiteMath::half

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

typedef struct Box4fT // LiteMath::Box4f, used by CrossRT (CRT_AABB)
{
  float4 boxMin;
  float4 boxMax;
} Box4f;

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

static inline int3 make_int3(int a, int b, int c)
{
  int3 res;
  res.x = a;
  res.y = b;
  res.z = c;
  return res;
}

static inline int4 make_int4(int a, int b, int c, int d)
{
  int4 res;
  res.x = a;
  res.y = b;
  res.z = c;
  res.w = d;
  return res;
}

static inline uint3 make_uint3(uint a, uint b, uint c)
{
  uint3 res;
  res.x = a;
  res.y = b;
  res.z = c;
  return res;
}

static inline uint4 make_uint4(uint a, uint b, uint c, uint d)
{
  uint4 res;
  res.x = a;
  res.y = b;
  res.z = c;
  res.w = d;
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

static inline float4 operator*(float4x4 m, float4 v) { return mul4x4x4(m, v); } // as LiteMath::operator*(float4x4, float4)

static inline float4x4 operator*(float4x4 m1, float4x4 m2)
{
  float4x4 res;
  res.m_col[0] = mul4x4x4(m1, m2.m_col[0]);
  res.m_col[1] = mul4x4x4(m1, m2.m_col[1]);
  res.m_col[2] = mul4x4x4(m1, m2.m_col[2]);
  res.m_col[3] = mul4x4x4(m1, m2.m_col[3]);
  return res;
}

static inline float4x4 rotate4x4X(float phi)
{
  float4x4 res;
  res.m_col[0] = make_float4(1.0f,      0.0f,     0.0f, 0.0f);
  res.m_col[1] = make_float4(0.0f, +cos(phi), +sin(phi), 0.0f);
  res.m_col[2] = make_float4(0.0f, -sin(phi), +cos(phi), 0.0f);
  res.m_col[3] = make_float4(0.0f,      0.0f,     0.0f, 1.0f);
  return res;
}

static inline float4x4 rotate4x4Y(float phi)
{
  float4x4 res;
  res.m_col[0] = make_float4(+cos(phi), 0.0f, -sin(phi), 0.0f);
  res.m_col[1] = make_float4(     0.0f, 1.0f,      0.0f, 0.0f);
  res.m_col[2] = make_float4(+sin(phi), 0.0f, +cos(phi), 0.0f);
  res.m_col[3] = make_float4(     0.0f, 0.0f,      0.0f, 1.0f);
  return res;
}

static inline float4x4 rotate4x4Z(float phi)
{
  float4x4 res;
  res.m_col[0] = make_float4(+cos(phi), sin(phi), 0.0f, 0.0f);
  res.m_col[1] = make_float4(-sin(phi), cos(phi), 0.0f, 0.0f);
  res.m_col[2] = make_float4(     0.0f,     0.0f, 1.0f, 0.0f);
  res.m_col[3] = make_float4(     0.0f,     0.0f, 0.0f, 1.0f);
  return res;
}

//static inline float3 reflect(const float3 dir, const float3 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }

static inline float  fract(float x) { return x - floor(x); }
//static inline float  clamp(float u, float a, float b) { return min(max(a, u), b); }
static inline float4 clamp(const float4 u, float a, float b) { return make_float4(clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b), clamp(u.w, a, b)); }
static inline float4 clamp(const float4 u, const float4 a, const float4 b) { return make_float4(clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z), clamp(u.w, a.w, b.w)); }
static inline float  lerp(const float a, const float b, float t)   { return a + t*(b - a); }
static inline float4 lerp(const float4 a, const float4 b, float t) { return a + t*(b - a); }

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

static inline float3 operator*(float3x3 m, float3 v) { return mul3x3x3(m, v); }

static inline float determinant(float3x3 m)
{
  return m.row[0].x*(m.row[1].y*m.row[2].z - m.row[1].z*m.row[2].y) - m.row[0].y*(m.row[1].x*m.row[2].z - m.row[1].z*m.row[2].x) + 
         m.row[0].z*(m.row[1].x*m.row[2].y - m.row[1].y*m.row[2].x);
}

/// the same formulas as LiteMath::inverse3x3 (element (r,c) is m.row[r].c)
static inline float3x3 inverse3x3(float3x3 m)
{
  const float invDet = 1.0f/determinant(m);
  const float a = m.row[0].x, b = m.row[0].y, c = m.row[0].z;
  const float d = m.row[1].x, e = m.row[1].y, f = m.row[1].z;
  const float g = m.row[2].x, h = m.row[2].y, i = m.row[2].z;
  float3x3 inv;
  inv.row[0] = make_float3((e*i - f*h)*invDet, (c*h - b*i)*invDet, (b*f - c*e)*invDet);
  inv.row[1] = make_float3((f*g - d*i)*invDet, (a*i - c*g)*invDet, (c*d - a*f)*invDet);
  inv.row[2] = make_float3((d*h - e*g)*invDet, (b*g - a*h)*invDet, (a*e - b*d)*invDet);
  return inv;
}

#define constexpr const static

static inline float dot (float4 a, float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
static inline float dot3(float4 a, float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
static inline float dot4(float4 a, float4 b)  { return dot(a,b); } 
static inline float dot (float3 a, float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }

static inline float dot (float2 a, float2 b)  { return a.x*b.x + a.y*b.y; }
static inline float  length(float2 a) { return sqrt(dot(a,a)); }
static inline float2 normalize(float2 a) { float lenInv = 1.0f/length(a); return a*lenInv; }
static inline float  length(float3 a) { return sqrt(dot(a,a)); }
static inline float3 normalize(float3 a) { float lenInv = 1.0f/length(a); return a*lenInv; }
static inline float  length(float4 a) { return sqrt(dot(a,a)); }
static inline float4 normalize(float4 a) { float lenInv = 1.0f/length(a); return a*lenInv; } // as LiteMath: 4 components
static inline float  length3(float4 a) { return sqrt(dot3(a,a)); }
static inline float4 normalize3(float4 a) { float lenInv = 1.0f/length3(a); return a*lenInv; }

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
static inline float4 cross3(float4 a, float4 b) { return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0f); } // w = 0 as in LiteMath
static inline float4 cross (float4 a, float4 b) { return cross3(a, b); }


//static inline float atomic_max_global(uniform float * uniform ptr, uniform float value) 
//{ 
//  uniform float cur;
//  while (value > (cur = *ptr))
//    value = atomic_compare_exchange_global(ptr, cur, value);
//  return value;
//}
//
//static inline float atomic_min_global(uniform float * uniform ptr, uniform float value) 
//{ 
//  uniform float cur;
//  while (value < (cur = *ptr))
//    value = atomic_compare_exchange_global(ptr, cur, value);
//  return value;
//}

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

/////////////////////////////////////////////////////////////////////
/////////////////// component-wise functions of LiteMath ////////////
/////////////////////////////////////////////////////////////////////

static inline float2 min(float2 a, float2 b) { return make_float2(min(a.x, b.x), min(a.y, b.y)); }
static inline float3 min(float3 a, float3 b) { return make_float3(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z)); }
static inline float4 min(float4 a, float4 b) { return make_float4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w)); }
static inline float2 max(float2 a, float2 b) { return make_float2(max(a.x, b.x), max(a.y, b.y)); }
static inline float3 max(float3 a, float3 b) { return make_float3(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)); }
static inline float4 max(float4 a, float4 b) { return make_float4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w)); }
static inline float2 abs(float2 a) { return make_float2(abs(a.x), abs(a.y)); }
static inline float3 abs(float3 a) { return make_float3(abs(a.x), abs(a.y), abs(a.z)); }
static inline float4 abs(float4 a) { return make_float4(abs(a.x), abs(a.y), abs(a.z), abs(a.w)); }
static inline float2 floor(float2 a) { return make_float2(floor(a.x), floor(a.y)); }
static inline float3 floor(float3 a) { return make_float3(floor(a.x), floor(a.y), floor(a.z)); }
static inline float4 floor(float4 a) { return make_float4(floor(a.x), floor(a.y), floor(a.z), floor(a.w)); }
static inline float2 sqrt(float2 a) { return make_float2(sqrt(a.x), sqrt(a.y)); }
static inline float3 sqrt(float3 a) { return make_float3(sqrt(a.x), sqrt(a.y), sqrt(a.z)); }
static inline float4 sqrt(float4 a) { return make_float4(sqrt(a.x), sqrt(a.y), sqrt(a.z), sqrt(a.w)); }
static inline float4 exp(float4 a) { return make_float4(exp(a.x), exp(a.y), exp(a.z), exp(a.w)); }
static inline float3 exp(float3 a) { return make_float3(exp(a.x), exp(a.y), exp(a.z)); }
static inline float4 pow(float4 a, float b) { return make_float4(pow(a.x, b), pow(a.y, b), pow(a.z, b), pow(a.w, b)); }
static inline float3 pow(float3 a, float b) { return make_float3(pow(a.x, b), pow(a.y, b), pow(a.z, b)); }
static inline float2 clamp(float2 u, float a, float b) { return make_float2(clamp(u.x, a, b), clamp(u.y, a, b)); }
static inline float3 clamp(float3 u, float a, float b) { return make_float3(clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)); }
static inline float2 clamp(float2 u, float2 a, float2 b) { return make_float2(clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y)); }
static inline float3 clamp(float3 u, float3 a, float3 b) { return make_float3(clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z)); }
static inline float2 lerp(float2 a, float2 b, float t) { return a + t*(b - a); }
static inline float3 lerp(float3 a, float3 b, float t) { return a + t*(b - a); }
static inline float  mix(float a, float b, float t)    { return a + t*(b - a); }
static inline float3 mix(float3 a, float3 b, float t)  { return a + t*(b - a); }
static inline float4 mix(float4 a, float4 b, float t)  { return a + t*(b - a); }
static inline float  sign(float x) { return (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f); }
static inline float  smoothstep(float edge0, float edge1, float x) { const float t = clamp((x - edge0)/(edge1 - edge0), 0.0f, 1.0f); return t*t*(3.0f - 2.0f*t); }
static inline float  hmax(float3 a) { return max(a.x, max(a.y, a.z)); }
static inline float  hmin(float3 a) { return min(a.x, min(a.y, a.z)); }
static inline float  hmax(float4 a) { return max(max(a.x, a.y), max(a.z, a.w)); }
static inline float  hmin(float4 a) { return min(min(a.x, a.y), min(a.z, a.w)); }
static inline float  hmax3(float4 a) { return max(a.x, max(a.y, a.z)); }
static inline float  hmin3(float4 a) { return min(a.x, min(a.y, a.z)); }
static inline float3 reflect(float3 dir, float3 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
static inline float4 reflect(float4 dir, float4 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
static inline float3 mul3x3(float4x4 m, float3 v) { return to_float3(m*to_float4(v, 0.0f)); }
static inline float3 mul4x3(float4x4 m, float3 v) { return to_float3(m*to_float4(v, 1.0f)); }
static inline float3 operator*(float4x4 m, float3 v) { return mul4x3(m, v); } // as LiteMath::operator*(float4x4, float3)
static inline float2 to_float2(float4 v) { return make_float2(v.x, v.y); }
static inline float2 to_float2(float3 v) { return make_float2(v.x, v.y); }
static inline uint   as_uint(float x)  { return intbits(x); }
static inline int    as_int(float x)   { return (int)intbits(x); }

/////////////////////////////////////////////////////////////////////
/////////////////// LiteMath::complex ///////////////////////////////
/////////////////////////////////////////////////////////////////////

struct complex { float re; float im; };

static inline complex make_complex(float re, float im) { complex res; res.re = re; res.im = im; return res; }
static inline complex make_complex(float re)           { return make_complex(re, 0.0f); }

static inline complex operator-(complex a)            { return make_complex(-a.re, -a.im); }
static inline complex operator+(complex a, complex b) { return make_complex(a.re + b.re, a.im + b.im); }
static inline complex operator-(complex a, complex b) { return make_complex(a.re - b.re, a.im - b.im); }
static inline complex operator*(complex a, complex b) { return make_complex(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re); }
static inline complex operator/(complex a, complex b)
{
  const float scale = 1.0f/(b.re*b.re + b.im*b.im);
  return make_complex(scale*(a.re*b.re + a.im*b.im), scale*(a.im*b.re - a.re*b.im));
}
static inline complex operator+(float a, complex b) { return make_complex(a) + b; } // implicit conversion of C++ 'complex(float)'
static inline complex operator-(float a, complex b) { return make_complex(a) - b; }
static inline complex operator*(float a, complex b) { return make_complex(a) * b; }
static inline complex operator/(float a, complex b) { return make_complex(a) / b; }
static inline complex operator+(complex a, float b) { return a + make_complex(b); }
static inline complex operator-(complex a, float b) { return a - make_complex(b); }
static inline complex operator*(complex a, float b) { return a * make_complex(b); }
static inline complex operator/(complex a, float b) { return a / make_complex(b); }

static inline float   real(complex z)         { return z.re; }
static inline float   imag(complex z)         { return z.im; }
static inline float   complex_norm(complex z) { return z.re*z.re + z.im*z.im; }
static inline float   complex_abs (complex z) { return sqrt(complex_norm(z)); }
static inline complex complex_sqrt(complex z)
{
  const float n  = complex_abs(z);
  const float t1 = sqrt(0.5f*(n + abs(z.re)));
  const float t2 = 0.5f*z.im/t1;
  if(n == 0.0f)
    return make_complex(0.0f, 0.0f);
  if(z.re >= 0.0f)
    return make_complex(t1, t2);
  return make_complex(abs(t2), (z.im < 0.0f) ? -t1 : t1); // copysign(t1, z.im)
}

#endif
