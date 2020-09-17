#ifndef VFLOAT4_ALL_H
#define VFLOAT4_ALL_H

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

  typedef unsigned int    uint;

  static inline int as_int(float x) 
  {
    int res; 
    memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
    return res; 
  }

  static inline uint as_uint(float x) 
  {
    uint res; 
    memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
    return res; 
  }

  static inline float as_float(int x)
  {
    float res; 
    memcpy(&res, &x, sizeof(float)); // modern C++ allow only this way, speed ik ok, check assembly with godbolt
    return res; 
  }

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
      struct {float x, y, z; };
      float M[3];
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

  static inline float clamp(float u, float a, float b) { const float r = fmax(a, u);      return fmin(r, b); }

  static inline int max  (int a, int b)        { return a > b ? a : b; }                                    
  static inline int min  (int a, int b)        { return a < b ? a : b; }                                    
  static inline int clamp(int u, int a, int b) { const int   r = (a > u) ? a : u; return (r < b) ? r : b; } 

  inline float rnd(float s, float e)
  {
    const float t = (float)(rand()) / (float)RAND_MAX;
    return s + t*(e - s);
  }

  template<typename T> inline T SQR(T x) { return x * x; }

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

  static inline float  lerp(float u, float v, float t) { return u + t * (v - u); }

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

  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static inline bool IntersectBox2Box2(float2 box1Min, float2 box1Max, float2 box2Min, float2 box2Max)
  {
    return box1Min.x <= box2Max.x && box2Min.x <= box1Max.x &&
           box1Min.y <= box2Max.y && box2Min.y <= box1Max.y;
  }

  static inline bool IntersectBox2Box2(int2 box1Min, int2 box1Max, int2 box2Min, int2 box2Max)
  {
    return box1Min.x <= box2Max.x && box2Min.x <= box1Max.x &&
           box1Min.y <= box2Max.y && box2Min.y <= box1Max.y;
  }

};

#endif
