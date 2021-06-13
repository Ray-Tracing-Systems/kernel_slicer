#ifndef OPENCL_MATH_CPU_H
#define OPENCL_MATH_CPU_H

#include <cmath>
#include <initializer_list>
#include <limits>

#include <cstring> // for memcpy

namespace LiteMath
{ 
  const float EPSILON      = 1e-6f;
  const float DEG_TO_RAD   = float(3.14159265358979323846f) / 180.0f;
  const float INF_POSITIVE = +std::numeric_limits<float>::infinity();
  const float INF_NEGATIVE = -std::numeric_limits<float>::infinity();

  template<typename T> inline T SQR(T x) { return x * x; }
  static inline float clamp(float u, float a, float b) { const float r = fmax(a, u); return fmin(r, b); }

  typedef unsigned int uint;

  static inline int as_int(float x) 
  {
    int res; 
    memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed is ok, check assembly with godbolt
    return res; 
  }

  static inline uint as_uint(float x) 
  {
    uint res; 
    memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed is ok, check assembly with godbolt
    return res; 
  }

  static inline float as_float(int x)
  {
    float res; 
    memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed is ok, check assembly with godbolt
    return res; 
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct float4
  {
    inline float4() : x(0), y(0), z(0), w(0) {}
    inline float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    inline explicit float4(float a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
    
    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }
    
    union
    {
      struct {float x, y, z, w; };
      float  M[4];
    };
  };

  static inline float4 operator * (const float4 & u, float v) { return float4(u.x * v, u.y * v, u.z * v, u.w * v); }
  static inline float4 operator / (const float4 & u, float v) { return float4(u.x / v, u.y / v, u.z / v, u.w / v); }
  static inline float4 operator + (const float4 & u, float v) { return float4(u.x + v, u.y + v, u.z + v, u.w + v); }
  static inline float4 operator - (const float4 & u, float v) { return float4(u.x - v, u.y - v, u.z - v, u.w - v); }
  static inline float4 operator * (float v, const float4 & u) { return float4(v * u.x, v * u.y, v * u.z, v * u.w); }
  static inline float4 operator / (float v, const float4 & u) { return float4(v / u.x, v / u.y, v / u.z, v / u.w); }
  static inline float4 operator + (float v, const float4 & u) { return float4(u.x + v, u.y + v, u.z + v, u.w + v); }
  static inline float4 operator - (float v, const float4 & u) { return float4(u.x - v, u.y - v, u.z - v, u.w - v); }

  static inline float4 operator + (const float4 & u, const float4 & v) { return float4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w); }
  static inline float4 operator - (const float4 & u, const float4 & v) { return float4(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w); }
  static inline float4 operator * (const float4 & u, const float4 & v) { return float4(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w); }
  static inline float4 operator / (const float4 & u, const float4 & v) { return float4(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w); }

  static inline float4 & operator += (float4 & u, const float4 & v) { u.x += v.x; u.y += v.y; u.z += v.z; u.w += v.w; return u; }
  static inline float4 & operator -= (float4 & u, const float4 & v) { u.x -= v.x; u.y -= v.y; u.z -= v.z; u.w -= v.w; return u; }
  static inline float4 & operator *= (float4 & u, const float4 & v) { u.x *= v.x; u.y *= v.y; u.z *= v.z; u.w *= v.w; return u; }
  static inline float4 & operator /= (float4 & u, const float4 & v) { u.x /= v.x; u.y /= v.y; u.z /= v.z; u.w /= v.w; return u; }

  static inline float4 & operator += (float4 & u, float v) { u.x += v; u.y += v; u.z += v; u.w += v; return u; }
  static inline float4 & operator -= (float4 & u, float v) { u.x -= v; u.y -= v; u.z -= v; u.w -= v; return u; }
  static inline float4 & operator *= (float4 & u, float v) { u.x *= v; u.y *= v; u.z *= v; u.w *= v; return u; }
  static inline float4 & operator /= (float4 & u, float v) { u.x /= v; u.y /= v; u.z /= v; u.w /= v; return u; }

  static inline float4   operator -(const float4 & v) { return float4(-v.x, -v.y, -v.z, -v.w); }

  static inline float4 lerp(const float4 & u, const float4 & v, float t) { return u + t * (v - u); }
  static inline float  dot(const float4 & u, const float4 & v) { return (u.x*v.x + u.y*v.y + u.z*v.z + u.w*v.w); }
  static inline float  dot3(const float4 & u, const float4 & v) { return (u.x*v.x + u.y*v.y + u.z*v.z); }
  static inline float4 cross(const float4 & u, const float4 & v) { return float4{u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x, u.w}; }
  static inline float  length3(const float4 & u) { return sqrtf(SQR(u.x) + SQR(u.y) + SQR(u.z)); }
  static inline float  length(const float4 & u) { return sqrtf(SQR(u.x) + SQR(u.y) + SQR(u.z) + SQR(u.w)); }
  static inline float4 clamp(const float4 & u, float a, float b) 
  { 
    return float4(clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b), clamp(u.w, a, b)); 
  }

  static inline float4 normalize(const float4 & u) { return u / length3(u); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct float3
  {
    inline float3() : x(0), y(0), z(0) {}
    inline float3(float a, float b, float c) : x(a), y(b), z(c) {}
    inline explicit float3(const float* ptr) : x(ptr[0]), y(ptr[1]), z(ptr[2]) {}

    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct {float x, y, z; }; // same aligment as for float4
      float M[4];               // same aligment as for float4
    };
  };

  struct float2
  {
    inline float2() : x(0), y(0) {}
    inline float2(float a, float b) : x(a), y(b) {}
    inline explicit float2(float a[2]) : x(a[0]), y(a[1]) {}

    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct {float x, y; };
      float M[2];
    };
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static inline int max  (int a, int b)        { return a > b ? a : b; }                                    
  static inline int min  (int a, int b)        { return a < b ? a : b; }                                    
  static inline int clamp(int u, int a, int b) { const int   r = (a > u) ? a : u; return (r < b) ? r : b; } 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static inline float2 to_float2(float3 v)          { return float2{v.x, v.y}; }

  //**********************************************************************************
  // float3 operators and functions
  //**********************************************************************************
  static inline float3 operator * (const float3 & u, float v) { return float3{u.x * v, u.y * v, u.z * v}; }
  static inline float3 operator / (const float3 & u, float v) { return float3{u.x / v, u.y / v, u.z / v}; }
  static inline float3 operator + (const float3 & u, float v) { return float3{u.x + v, u.y + v, u.z + v}; }
  static inline float3 operator - (const float3 & u, float v) { return float3{u.x - v, u.y - v, u.z - v}; }
  static inline float3 operator * (float v, const float3 & u) { return float3{v * u.x, v * u.y, v * u.z}; }
  static inline float3 operator / (float v, const float3 & u) { return float3{v / u.x, v / u.y, v / u.z}; }
  static inline float3 operator + (float v, const float3 & u) { return float3{u.x + v, u.y + v, u.z + v}; }
  static inline float3 operator - (float v, const float3 & u) { return float3{u.x - v, u.y - v, u.z - v}; }

  static inline float3 operator + (const float3 & u, const float3 & v) { return float3{u.x + v.x, u.y + v.y, u.z + v.z}; }
  static inline float3 operator - (const float3 & u, const float3 & v) { return float3{u.x - v.x, u.y - v.y, u.z - v.z}; }
  static inline float3 operator * (const float3 & u, const float3 & v) { return float3{u.x * v.x, u.y * v.y, u.z * v.z}; }
  static inline float3 operator / (const float3 & u, const float3 & v) { return float3{u.x / v.x, u.y / v.y, u.z / v.z}; }

  static inline float3 operator - (const float3 & u) { return {-u.x, -u.y, -u.z}; }

  static inline float3 & operator += (float3 & u, const float3 & v) { u.x += v.x; u.y += v.y; u.z += v.z; return u; }
  static inline float3 & operator -= (float3 & u, const float3 & v) { u.x -= v.x; u.y -= v.y; u.z -= v.z; return u; }
  static inline float3 & operator *= (float3 & u, const float3 & v) { u.x *= v.x; u.y *= v.y; u.z *= v.z; return u; }
  static inline float3 & operator /= (float3 & u, const float3 & v) { u.x /= v.x; u.y /= v.y; u.z /= v.z; return u; }

  static inline float3 & operator += (float3 & u, float v) { u.x += v; u.y += v; u.z += v; return u; }
  static inline float3 & operator -= (float3 & u, float v) { u.x -= v; u.y -= v; u.z -= v; return u; }
  static inline float3 & operator *= (float3 & u, float v) { u.x *= v; u.y *= v; u.z *= v; return u; }
  static inline float3 & operator /= (float3 & u, float v) { u.x /= v; u.y /= v; u.z /= v; return u; }
  static inline bool     operator == (const float3 & u, const float3 & v) { return (::fabs(u.x - v.x) < EPSILON) && (::fabs(u.y - v.y) < EPSILON) && (::fabs(u.z - v.z) < EPSILON); }
  
  static inline float3 lerp(const float3 & u, const float3 & v, float t) { return u + t * (v - u); }
  static inline float  dot(const float3 & u, const float3 & v) { return (u.x*v.x + u.y*v.y + u.z*v.z); }
  static inline float3 cross(const float3 & u, const float3 & v) { return float3{u.y*v.z - u.z*v.y, u.z*v.x - u.x*v.z, u.x*v.y - u.y*v.x}; }
  static inline float3 clamp(const float3 & u, float a, float b) { return float3{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)}; }
  static inline float3 min  (const float3& a, const float3& b)  { return float3{fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)}; }
  static inline float3 max  (const float3& a, const float3& b)  { return float3{fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)}; }

  static inline float  length(const float3 & u) { return sqrtf(SQR(u.x) + SQR(u.y) + SQR(u.z)); }
  static inline float  lengthSquare(const float3 u) { return u.x*u.x + u.y*u.y + u.z*u.z; }
  static inline float3 normalize(const float3 & u) { return u / length(u); }

  static inline float  maxcomp(const float3 & u) { return fmax(u.x, fmax(u.y, u.z)); }
  static inline float  mincomp(const float3 & u) { return fmin(u.x, fmin(u.y, u.z)); }

  //**********************************************************************************
  // float2 operators and functions
  //**********************************************************************************

  static inline float2 operator * (const float2 & u, float v) { return float2{u.x * v, u.y * v}; }
  static inline float2 operator / (const float2 & u, float v) { return float2{u.x / v, u.y / v}; }
  static inline float2 operator * (float v, const float2 & u) { return float2{v * u.x, v * u.y}; }
  static inline float2 operator / (float v, const float2 & u) { return float2{v / u.x, v / u.y}; }

  static inline float2 operator + (const float2 & u, const float2 & v) { return float2{u.x + v.x, u.y + v.y}; }
  static inline float2 operator - (const float2 & u, const float2 & v) { return float2{u.x - v.x, u.y - v.y}; }
  static inline float2 operator * (const float2 & u, const float2 & v) { return float2{u.x * v.x, u.y * v.y}; }
  static inline float2 operator / (const float2 & u, const float2 & v) { return float2{u.x / v.x, u.y / v.y}; }
  static inline float2 operator - (const float2 & v) { return {-v.x, -v.y}; }

  static inline float2 & operator += (float2 & u, const float2 & v) { u.x += v.x; u.y += v.y; return u; }
  static inline float2 & operator -= (float2 & u, const float2 & v) { u.x -= v.x; u.y -= v.y; return u; }
  static inline float2 & operator *= (float2 & u, const float2 & v) { u.x *= v.x; u.y *= v.y; return u; }
  static inline float2 & operator /= (float2 & u, const float2 & v) { u.x /= v.x; u.y /= v.y; return u; }

  static inline float2 & operator += (float2 & u, float v) { u.x += v; u.y += v; return u; }
  static inline float2 & operator -= (float2 & u, float v) { u.x -= v; u.y -= v; return u; }
  static inline float2 & operator *= (float2 & u, float v) { u.x *= v; u.y *= v; return u; }
  static inline float2 & operator /= (float2 & u, float v) { u.x /= v; u.y /= v; return u; }
  static inline bool     operator == (const float2 & u, const float2 & v) { return (::fabs(u.x - v.x) < EPSILON) && (::fabs(u.y - v.y) < EPSILON); }

  static inline float2 lerp(const float2 & u, const float2 & v, float t) { return u + t * (v - u); }
  static inline float  dot(const float2 & u, const float2 & v)   { return (u.x*v.x + u.y*v.y); }
  static inline float2 clamp(const float2 & u, float a, float b) { return float2{clamp(u.x, a, b), clamp(u.y, a, b)}; }

  static inline float  length(const float2 & u)    { return sqrtf(SQR(u.x) + SQR(u.y)); }
  static inline float2 normalize(const float2 & u) { return u / length(u); }
  static inline float2 floor(float2 v) { return float2(floorf(v.x), floorf(v.y)); }
  static inline float  lerp(float u, float v, float t) { return u + t * (v - u); }

  static inline float smoothstep(float edge0, float edge1, float x)
  {
    float  tVal = (x - edge0) / (edge1 - edge0);
    float  t    = fmin(fmax(tVal, 0.0f), 1.0f); 
    return t * t * (3.0f - 2.0f * t);
  }

  static inline float mix(float x, float y, float a)
  {
   return x*(1.0f - a) + y*a;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct int3
  {
      int3() :x(0), y(0), z(0) {}
      int3(int a, int b, int c) : x(a), y(b), z(c) {}
      inline int& operator[](int i)       { return M[i]; }
      inline int  operator[](int i) const { return M[i]; }

      union
      {
          struct {int x, y, z; };
          int M[3];
      };
  };

  static inline int3 operator ^ (const int3 & u, const int3 & v) { return int3{u.x ^ v.x, u.y ^ v.y, u.z ^ v.z}; }
  static inline int3 & operator ^= (int3 & u, const int3 & v) { u.x ^= v.x; u.y ^= v.y; u.z ^= v.z; return u; }

  struct uint3
  {
      uint3() :x(0), y(0), z(0) {}
      uint3(unsigned int a, unsigned int b, unsigned int c) : x(a), y(b), z(c) {}

      unsigned int x, y, z;
  };

  struct int2
  {
    int2() : x(0), y(0) {}
    int2(int a, int b) : x(a), y(b) {}

    int x, y;
  };

  struct uint2
  {
    uint2() : x(0), y(0) {}
    uint2(unsigned int a, unsigned int b) : x(a), y(b) {}

    bool operator==(const uint2 &other) const { return (x == other.x && y == other.y) || (x == other.y && y == other.x); }

    unsigned int x, y;
  };

  struct uint4
  {
    uint4() : x(0), y(0), z(0), w(0) {}
    uint4(unsigned int a, unsigned int b, unsigned int c, unsigned int d) : x(a), y(b), z(c), w(d) {}

    unsigned int x,y,z,w;
  };

  struct ushort2
  {
    ushort2() : x(0), y(0) {}
    ushort2(unsigned short a, unsigned short b) : x(a), y(b) {}

    unsigned short x, y;
  };

  struct ushort4
  {
    ushort4() :x(0), y(0), z(0), w(0) {}
    ushort4(unsigned short a, unsigned short b, unsigned short c, unsigned short d) : x(a), y(b), z(c), w(d) {}

    unsigned short x, y, z, w;
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct uchar4
  {
    inline uchar4() : x(0), y(0), z(0), w(0) {}
    inline uchar4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) : x(a), y(b), z(c), w(d) {}
    inline explicit uchar4(unsigned char a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
    
    inline float& operator[](uint i)       { return M[i]; }
    inline float  operator[](uint i) const { return M[i]; }
    
    union
    {
      struct {unsigned char x, y, z, w; };
      float  M[4];
    };
  };

  static inline uchar4 operator * (const uchar4 & u, float v) { return uchar4(u.x * v, u.y * v, u.z * v, u.w * v); }
  static inline uchar4 operator / (const uchar4 & u, float v) { return uchar4(u.x / v, u.y / v, u.z / v, u.w / v); }
  static inline uchar4 operator + (const uchar4 & u, float v) { return uchar4(u.x + v, u.y + v, u.z + v, u.w + v); }
  static inline uchar4 operator - (const uchar4 & u, float v) { return uchar4(u.x - v, u.y - v, u.z - v, u.w - v); }
  static inline uchar4 operator * (float v, const uchar4 & u) { return uchar4(v * u.x, v * u.y, v * u.z, v * u.w); }
  static inline uchar4 operator / (float v, const uchar4 & u) { return uchar4(v / u.x, v / u.y, v / u.z, v / u.w); }
  static inline uchar4 operator + (float v, const uchar4 & u) { return uchar4(u.x + v, u.y + v, u.z + v, u.w + v); }
  static inline uchar4 operator - (float v, const uchar4 & u) { return uchar4(u.x - v, u.y - v, u.z - v, u.w - v); }

  static inline uchar4 operator + (const uchar4 & u, const uchar4 & v) { return uchar4(u.x + v.x, u.y + v.y, u.z + v.z, u.w + v.w); }
  static inline uchar4 operator - (const uchar4 & u, const uchar4 & v) { return uchar4(u.x - v.x, u.y - v.y, u.z - v.z, u.w - v.w); }
  static inline uchar4 operator * (const uchar4 & u, const uchar4 & v) { return uchar4(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w); }
  static inline uchar4 operator / (const uchar4 & u, const uchar4 & v) { return uchar4(u.x / v.x, u.y / v.y, u.z / v.z, u.w / v.w); }

  static inline uchar4 lerp(const uchar4 & u, const uchar4 & v, float t) { return u + t * (v - u); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  
  struct float4x4
  {
    inline float4x4()  { identity(); }

    inline explicit float4x4(const float A[16])
    {
      m_col[0] = float4{ A[0], A[4], A[8],  A[12] };
      m_col[1] = float4{ A[1], A[5], A[9],  A[13] };
      m_col[2] = float4{ A[2], A[6], A[10], A[14] };
      m_col[3] = float4{ A[3], A[7], A[11], A[15] };
    }

    inline void identity()
    {
      m_col[0] = float4{ 1.0f, 0.0f, 0.0f, 0.0f };
      m_col[1] = float4{ 0.0f, 1.0f, 0.0f, 0.0f };
      m_col[2] = float4{ 0.0f, 0.0f, 1.0f, 0.0f };
      m_col[3] = float4{ 0.0f, 0.0f, 0.0f, 1.0f };
    }

    inline float4 get_col(int i) const                { return m_col[i]; }
    inline void   set_col(int i, const float4& a_col) { m_col[i] = a_col; }

    inline float4 get_row(int i) const { return float4{ m_col[0][i], m_col[1][i], m_col[2][i], m_col[3][i] }; }
    inline void   set_row(int i, const float4& a_col)
    {
      m_col[0][i] = a_col[0];
      m_col[1][i] = a_col[1];
      m_col[2][i] = a_col[2];
      m_col[3][i] = a_col[3];
    }

    inline float4& col(int i)       { return m_col[i]; }
    inline float4  col(int i) const { return m_col[i]; }

    inline float& operator()(int row, int col)       { return m_col[col][row]; }
    inline float  operator()(int row, int col) const { return m_col[col][row]; }

    float4 m_col[4];
  };


  static inline float4 mul(float4x4 m, float4 v)
  {
    float4 res;
    res.x = m.get_row(0).x*v.x + m.get_row(0).y*v.y + m.get_row(0).z*v.z + m.get_row(0).w*v.w;
    res.y = m.get_row(1).x*v.x + m.get_row(1).y*v.y + m.get_row(1).z*v.z + m.get_row(1).w*v.w;
    res.z = m.get_row(2).x*v.x + m.get_row(2).y*v.y + m.get_row(2).z*v.z + m.get_row(2).w*v.w;
    res.w = m.get_row(3).x*v.x + m.get_row(3).y*v.y + m.get_row(3).z*v.z + m.get_row(3).w*v.w;
    return res;
  }

  static inline float4x4 mul(float4x4 m1, float4x4 m2)
  {
    const float4 column1 = mul(m1, m2.col(0));
    const float4 column2 = mul(m1, m2.col(1));
    const float4 column3 = mul(m1, m2.col(2));
    const float4 column4 = mul(m1, m2.col(3));
    float4x4 res;
    res.set_col(0, column1);
    res.set_col(1, column2);
    res.set_col(2, column3);
    res.set_col(3, column4);

    return res;
  }

  static inline float4x4 inverse4x4(float4x4 m1)
  {
    float tmp[12]; // temp array for pairs
    float4x4 m;

    // calculate pairs for first 8 elements (cofactors)
    //
    tmp[0]  = m1(2,2) * m1(3,3);
    tmp[1]  = m1(2,3) * m1(3,2);
    tmp[2]  = m1(2,1) * m1(3,3);
    tmp[3]  = m1(2,3) * m1(3,1);
    tmp[4]  = m1(2,1) * m1(3,2);
    tmp[5]  = m1(2,2) * m1(3,1);
    tmp[6]  = m1(2,0) * m1(3,3);
    tmp[7]  = m1(2,3) * m1(3,0);
    tmp[8]  = m1(2,0) * m1(3,2);
    tmp[9]  = m1(2,2) * m1(3,0);
    tmp[10] = m1(2,0) * m1(3,1);
    tmp[11] = m1(2,1) * m1(3,0);

    // calculate first 8 m1.rowents (cofactors)
    //
    m(0,0) = tmp[0]  * m1(1,1) + tmp[3] * m1(1,2) + tmp[4]  * m1(1,3);
    m(0,0) -= tmp[1] * m1(1,1) + tmp[2] * m1(1,2) + tmp[5]  * m1(1,3);
    m(1,0) = tmp[1]  * m1(1,0) + tmp[6] * m1(1,2) + tmp[9]  * m1(1,3);
    m(1,0) -= tmp[0] * m1(1,0) + tmp[7] * m1(1,2) + tmp[8]  * m1(1,3);
    m(2,0) = tmp[2]  * m1(1,0) + tmp[7] * m1(1,1) + tmp[10] * m1(1,3);
    m(2,0) -= tmp[3] * m1(1,0) + tmp[6] * m1(1,1) + tmp[11] * m1(1,3);
    m(3,0) = tmp[5]  * m1(1,0) + tmp[8] * m1(1,1) + tmp[11] * m1(1,2);
    m(3,0) -= tmp[4] * m1(1,0) + tmp[9] * m1(1,1) + tmp[10] * m1(1,2);
    m(0,1) = tmp[1]  * m1(0,1) + tmp[2] * m1(0,2) + tmp[5]  * m1(0,3);
    m(0,1) -= tmp[0] * m1(0,1) + tmp[3] * m1(0,2) + tmp[4]  * m1(0,3);
    m(1,1) = tmp[0]  * m1(0,0) + tmp[7] * m1(0,2) + tmp[8]  * m1(0,3);
    m(1,1) -= tmp[1] * m1(0,0) + tmp[6] * m1(0,2) + tmp[9]  * m1(0,3);
    m(2,1) = tmp[3]  * m1(0,0) + tmp[6] * m1(0,1) + tmp[11] * m1(0,3);
    m(2,1) -= tmp[2] * m1(0,0) + tmp[7] * m1(0,1) + tmp[10] * m1(0,3);
    m(3,1) = tmp[4]  * m1(0,0) + tmp[9] * m1(0,1) + tmp[10] * m1(0,2);
    m(3,1) -= tmp[5] * m1(0,0) + tmp[8] * m1(0,1) + tmp[11] * m1(0,2);

    // calculate pairs for second 8 m1.rowents (cofactors)
    //
    tmp[0]  = m1(0,2) * m1(1,3);
    tmp[1]  = m1(0,3) * m1(1,2);
    tmp[2]  = m1(0,1) * m1(1,3);
    tmp[3]  = m1(0,3) * m1(1,1);
    tmp[4]  = m1(0,1) * m1(1,2);
    tmp[5]  = m1(0,2) * m1(1,1);
    tmp[6]  = m1(0,0) * m1(1,3);
    tmp[7]  = m1(0,3) * m1(1,0);
    tmp[8]  = m1(0,0) * m1(1,2);
    tmp[9]  = m1(0,2) * m1(1,0);
    tmp[10] = m1(0,0) * m1(1,1);
    tmp[11] = m1(0,1) * m1(1,0);

    // calculate second 8 m1 (cofactors)
    //
    m(0,2) = tmp[0]   * m1(3,1) + tmp[3]  * m1(3,2) + tmp[4]  * m1(3,3);
    m(0,2) -= tmp[1]  * m1(3,1) + tmp[2]  * m1(3,2) + tmp[5]  * m1(3,3);
    m(1,2) = tmp[1]   * m1(3,0) + tmp[6]  * m1(3,2) + tmp[9]  * m1(3,3);
    m(1,2) -= tmp[0]  * m1(3,0) + tmp[7]  * m1(3,2) + tmp[8]  * m1(3,3);
    m(2,2) = tmp[2]   * m1(3,0) + tmp[7]  * m1(3,1) + tmp[10] * m1(3,3);
    m(2,2) -= tmp[3]  * m1(3,0) + tmp[6]  * m1(3,1) + tmp[11] * m1(3,3);
    m(3,2) = tmp[5]   * m1(3,0) + tmp[8]  * m1(3,1) + tmp[11] * m1(3,2);
    m(3,2) -= tmp[4]  * m1(3,0) + tmp[9]  * m1(3,1) + tmp[10] * m1(3,2);
    m(0,3) = tmp[2]   * m1(2,2) + tmp[5]  * m1(2,3) + tmp[1]  * m1(2,1);
    m(0,3) -= tmp[4]  * m1(2,3) + tmp[0]  * m1(2,1) + tmp[3]  * m1(2,2);
    m(1,3) = tmp[8]   * m1(2,3) + tmp[0]  * m1(2,0) + tmp[7]  * m1(2,2);
    m(1,3) -= tmp[6]  * m1(2,2) + tmp[9]  * m1(2,3) + tmp[1]  * m1(2,0);
    m(2,3) = tmp[6]   * m1(2,1) + tmp[11] * m1(2,3) + tmp[3]  * m1(2,0);
    m(2,3) -= tmp[10] * m1(2,3) + tmp[2]  * m1(2,0) + tmp[7]  * m1(2,1);
    m(3,3) = tmp[10]  * m1(2,2) + tmp[4]  * m1(2,0) + tmp[9]  * m1(2,1);
    m(3,3) -= tmp[8]  * m1(2,1) + tmp[11] * m1(2,2) + tmp[5]  * m1(2,0);

    // calculate matrix inverse
    //
    const float k = 1.0f / (m1(0,0) * m(0,0) + m1(0,1) * m(1,0) + m1(0,2) * m(2,0) + m1(0,3) * m(3,0));
    const float4 vK{k,k,k,k};

    m.set_col(0, m.get_col(0)*vK);
    m.set_col(1, m.get_col(1)*vK);
    m.set_col(2, m.get_col(2)*vK);
    m.set_col(3, m.get_col(3)*vK);

    return m;
  }

  static inline float4x4 transpose(const float4x4& rhs)
  {
    float4x4 res;
    for(int i=0;i<4;i++)
      res.set_row(i,rhs.get_col(i));
    return res;
  }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

};

#endif
