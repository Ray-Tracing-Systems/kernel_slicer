#ifndef VFLOAT4_ALL_H
#define VFLOAT4_ALL_H

//#include "vfloat4_gcc.h"

#ifdef WIN32
  #include "vfloat4_x64.h"
#else
  #if defined(__arm__) or defined(__aarch64__)
    #include "vfloat4_arm.h"
  #elif defined(__GNUC__)
    #include "vfloat4_gcc.h"
  #elif defined(__clang__)
    #include "vfloat4_x64.h"
  #else
    #include "vfloat4_x64.h"
  #endif  
#endif

// This is just and example. 
// In practise you may take any of these files that you prefer for your platform.  
// Or may use one of them or make yourself impl

//#include "vfloat4_gcc.h"
//#include "vfloat4_x64.h"
//#include "vfloat4_arm.h"

// __mips__
// __ppc__ 

#include <cmath>
#include <initializer_list>
#include <limits>

#include <cstring> // for memcpy

namespace LiteMath
{ 
  constexpr float EPSILON      = 1e-6f;
  constexpr float DEG_TO_RAD   = float(3.14159265358979323846f) / 180.0f;
  constexpr float INF_POSITIVE = +std::numeric_limits<float>::infinity();
  constexpr float INF_NEGATIVE = -std::numeric_limits<float>::infinity();

  typedef unsigned int uint;


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

  static inline float clamp(float u, float a, float b) { return std::min(std::max(a, u), b); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct uint4
  {
    inline uint4()                           : x(0), y(0), z(0), w(0) {}
    inline uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
    inline explicit uint4(uint a[4])          : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline uint4(const std::initializer_list<uint> a_v) { v = cvex::load_u(a_v.begin()); }
    inline uint4(cvex::vuint4 rhs) { v = rhs; }
    inline uint4 operator=(cvex::vuint4 rhs) { v = rhs; return *this; }

    inline uint& operator[](int i)       { return M[i]; }
    inline uint  operator[](int i) const { return M[i]; }

    inline uint4 operator+(const uint4& b) const { return v + b.v; }
    inline uint4 operator-(const uint4& b) const { return v - b.v; }
    inline uint4 operator*(const uint4& b) const { return v * b.v; }
    inline uint4 operator/(const uint4& b) const { return v / b.v; }

    inline uint4 operator+(const uint rhs) const { return v + rhs; }
    inline uint4 operator-(const uint rhs) const { return v - rhs; }
    inline uint4 operator*(const uint rhs) const { return v * rhs; }
    inline uint4 operator/(const uint rhs) const { return v / rhs; }

    inline void operator*=(const uint rhs) { v = v * rhs; }
    inline void operator*=(const uint4& b) { v = v * b.v; }

    inline void operator/=(const uint rhs) { v = v / rhs; }
    inline void operator/=(const uint4& b) { v = v / b.v; }

    inline void operator+=(const uint b  ) { v = v + b;   }
    inline void operator+=(const uint4& b) { v = v + b.v; }

    inline void operator-=(const uint   b) { v = v - b;   }
    inline void operator-=(const uint4& b) { v = v - b.v; }

    inline uint4 operator> (const uint4& b) const { return (v > b.v); }
    inline uint4 operator< (const uint4& b) const { return (v < b.v); }
    inline uint4 operator>=(const uint4& b) const { return (v >= b.v); }
    inline uint4 operator<=(const uint4& b) const { return (v <= b.v); }
    inline uint4 operator==(const uint4& b) const { return (v == b.v); }
    inline uint4 operator!=(const uint4& b) const { return (v != b.v); }

    union
    {
      struct { uint x, y, z, w; };
      uint M[4];
      cvex::vuint4 v;
    };
  };

  static inline uint4 operator+(const uint a, const uint4 b) 
  { 
    const cvex::vuint4 res = (a + b.v);
    return uint4(res); 
  }

  static inline uint4 operator-(const uint a, const uint4 b) 
  { 
    const cvex::vuint4 res = (a - b.v);
    return uint4(res); 
  }

  static inline uint4 operator*(const uint a, const uint4 b) 
  { 
    const cvex::vuint4 res = (a * b.v);
    return uint4(res); 
  }

  static inline uint4 operator/(const uint a, const uint4 b) 
  { 
    const cvex::vuint4 res = (a / b.v);
    return uint4(res); 
  }

  static inline uint4 load   (const uint* p)       { return cvex::load(p);      }
  static inline uint4 load_u (const uint* p)       { return cvex::load_u(p);    }
  static inline void  store  (uint* p, uint4 a_val) { cvex::store  (p, a_val.v); }
  static inline void  store_u(uint* p, uint4 a_val) { cvex::store_u(p, a_val.v); }

  static inline uint4 operator&(const uint4 a, const uint4 b) { return uint4(a.v & b.v); }
  static inline uint4 operator|(const uint4 a, const uint4 b) { return uint4(a.v | b.v); }
  static inline uint4 operator~(const uint4 a)                { return uint4(~a.v); }

  static inline uint4 operator>>(const uint4 a, const uint b) { return uint4(a.v >> b); }
  static inline uint4 operator<<(const uint4 a, const uint b) { return uint4(a.v << b); }
  
  static inline uint4 min  (const uint4& a,   const uint4& b) { return cvex::min(a.v, b.v); }
  static inline uint4 max  (const uint4& a,   const uint4& b) { return cvex::max(a.v, b.v); }
  static inline uint4 clamp(const uint4& a_x, const uint4& a_min, const uint4& a_max) { return cvex::clamp(a_x.v, a_min.v, a_max.v); }
  static inline uint4 clamp(const uint4& u, uint a, uint b)                           { return cvex::clamp(u.v, cvex::splat(a), cvex::splat(b)); }

  static inline bool any_of (const uint4 a)  { return cvex::any_of(a.v); }
  static inline bool all_of (const uint4 a)  { return cvex::all_of(a.v); }

  static inline uint4 blend(const uint4 a, const uint4 b, const uint4 mask) { return cvex::blend(a.v, b.v, mask.v); }

  static inline uint4 shuffle_xzyw(uint4 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline uint4 shuffle_yxzw(uint4 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline uint4 shuffle_yzxw(uint4 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline uint4 shuffle_zxyw(uint4 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline uint4 shuffle_zyxw(uint4 a_src) { return cvex::shuffle_zyxw(a_src.v); }
  static inline uint4 shuffle_xyxy(uint4 a_src) { return cvex::shuffle_xyxy(a_src.v); }
  static inline uint4 shuffle_zwzw(uint4 a_src) { return cvex::shuffle_zwzw(a_src.v); }

  static inline uint extract_0(const uint4& a_val) { return cvex::extract_0(a_val.v); }
  static inline uint extract_1(const uint4& a_val) { return cvex::extract_1(a_val.v); }
  static inline uint extract_2(const uint4& a_val) { return cvex::extract_2(a_val.v); }
  static inline uint extract_3(const uint4& a_val) { return cvex::extract_3(a_val.v); }

  static inline uint4 splat_0(const uint4& v)   { return cvex::splat_0(v.v); }
  static inline uint4 splat_1(const uint4& v)   { return cvex::splat_1(v.v); }
  static inline uint4 splat_2(const uint4& v)   { return cvex::splat_2(v.v); }
  static inline uint4 splat_3(const uint4& v)   { return cvex::splat_3(v.v); }  
  static inline uint4 splat  (const uint s)     { return cvex::splat(s); }  

  static inline uint  hmin3(const uint4 a_val) { return cvex::hmin3(a_val.v); }
  static inline uint  hmax3(const uint4 a_val) { return cvex::hmax3(a_val.v); } 
  static inline uint  hmin(const uint4 a_val)  { return cvex::hmin(a_val.v); }
  static inline uint  hmax(const uint4 a_val)  { return cvex::hmax(a_val.v); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct int4
  {
    inline int4()                           : x(0), y(0), z(0), w(0) {}
    inline int4(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {}
    inline explicit int4(int a[4])          : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline int4(const std::initializer_list<int> a_v) { v = cvex::load_u(a_v.begin()); }
    inline int4(cvex::vint4 rhs) { v = rhs; }
    inline int4 operator=(cvex::vint4 rhs) { v = rhs; return *this; }

    inline int& operator[](int i)       { return M[i]; }
    inline int  operator[](int i) const { return M[i]; }

    inline int4 operator+(const int4& b) const { return v + b.v; }
    inline int4 operator-(const int4& b) const { return v - b.v; }
    inline int4 operator*(const int4& b) const { return v * b.v; }
    inline int4 operator/(const int4& b) const { return v / b.v; }

    inline int4 operator+(const int rhs) const { return v + rhs; }
    inline int4 operator-(const int rhs) const { return v - rhs; }
    inline int4 operator*(const int rhs) const { return v * rhs; }
    inline int4 operator/(const int rhs) const { return v / rhs; }

    union
    {
      struct { int x, y, z, w; };
      int  M[4];
      cvex::vint4 v;
    };
  };

  static inline int4 operator+(const int a, const int4 b) 
  { 
    const cvex::vint4 res = (a + b.v);
    return int4(res); 
  }

  static inline int4 operator-(const int a, const int4 b) 
  { 
    const cvex::vint4 res = (a - b.v);
    return int4(res); 
  }

  static inline int4 operator*(const int a, const int4 b) 
  { 
    const cvex::vint4 res = (a * b.v);
    return int4(res); 
  }

  static inline int4 operator/(const int a, const int4 b) 
  { 
    const cvex::vint4 res = (a / b.v);
    return int4(res); 
  }

  static inline int4 load   (const int* p)       { return cvex::load(p);      }
  static inline int4 load_u (const int* p)       { return cvex::load_u(p);    }
  static inline void store  (int* p, int4 a_val) { cvex::store  (p, a_val.v); }
  static inline void store_u(int* p, int4 a_val) { cvex::store_u(p, a_val.v); }

  static inline int4 operator&(const int4 a, const int4 b) { return int4(a.v & b.v); }
  static inline int4 operator|(const int4 a, const int4 b) { return int4(a.v | b.v); }
  static inline int4 operator~(const int4 a)               { return int4(~a.v); }

  static inline int4 operator>>(const int4 a, const int b) { return int4(a.v >> b); }
  static inline int4 operator<<(const int4 a, const int b) { return int4(a.v << b); }

  static inline int4 min  (const int4& a,   const int4& b)                        { return int4( cvex::min(a.v, b.v) ); }
  static inline int4 max  (const int4& a,   const int4& b)                        { return int4( cvex::max(a.v, b.v) ); }
  static inline int4 clamp(const int4& a_x, const int4& a_min, const int4& a_max) { return cvex::clamp(a_x.v, a_min.v, a_max.v); }
  static inline int4 clamp(const int4& a_x, const int a_min, const int a_max)     { return cvex::clamp(a_x.v, cvex::splat(a_min), cvex::splat(a_max)); }

  static inline bool any_of (const int4 a) { return cvex::any_of(a.v); }
  static inline bool all_of (const int4 a) { return cvex::all_of(a.v); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct float4
  {
    inline float4() : x(0), y(0), z(0), w(0) {}
    inline float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    inline explicit float4(float a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline float4(const std::initializer_list<float> a_v) { v = cvex::load_u(a_v.begin()); }
    inline float4          (cvex::vfloat4 rhs)            { v = rhs; }
    inline float4 operator=(cvex::vfloat4 rhs)            { v = rhs; return *this; }
    
    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    inline float4 operator+(const float4& b) const { return v + b.v; }
    inline float4 operator-(const float4& b) const { return v - b.v; }
    inline float4 operator*(const float4& b) const { return v * b.v; }
    inline float4 operator/(const float4& b) const { return v / b.v; }

    inline float4 operator+(const float rhs) const { return v + rhs; }
    inline float4 operator-(const float rhs) const { return v - rhs; }
    inline float4 operator*(const float rhs) const { return v * rhs; }
    inline float4 operator/(const float rhs) const { return v / rhs; }

    inline void operator*=(const float rhs) { v = v * rhs; }
    inline void operator*=(const float4& b) { v = v * b.v; }

    inline void operator/=(const float rhs) { v = v / rhs; }
    inline void operator/=(const float4& b) { v = v / b.v; }

    inline void operator+=(const float4& b) { v = v + b.v; }
    inline void operator+=(const float b  ) { v = v + b;   }

    inline void operator-=(const float4& b) { v = v - b.v; }
    inline void operator-=(const float   b) { v = v - b;   }

    inline uint4 operator> (const float4& b) const { return (v > b.v); }
    inline uint4 operator< (const float4& b) const { return (v < b.v); }
    inline uint4 operator>=(const float4& b) const { return (v >= b.v); }
    inline uint4 operator<=(const float4& b) const { return (v <= b.v); }
    inline uint4 operator==(const float4& b) const { return (v == b.v); }
    inline uint4 operator!=(const float4& b) const { return (v != b.v); }

    union
    {
      struct {float x, y, z, w; };
      float  M[4];
      cvex::vfloat4 v;
    };
  };

  static inline float4 operator+(const float a, const float4 b) 
  { 
    const cvex::vfloat4 res = (a + b.v);
    return float4(res); 
  }

  static inline float4 operator-(const float a, const float4 b) 
  { 
    const cvex::vfloat4 res = (a - b.v);
    return float4(res); 
  }

  static inline float4 operator*(const float a, const float4 b) 
  { 
    const cvex::vfloat4 res = (a * b.v);
    return float4(res); 
  }

  static inline float4 operator/(const float a, const float4 b) 
  { 
    const cvex::vfloat4 res = (a / b.v);
    return float4(res); 
  }

  static inline float4 load   (const float* p)         { return cvex::load(p);      }
  static inline float4 load_u (const float* p)         { return cvex::load_u(p);    }
  static inline void   store  (float* p, float4 a_val) { cvex::store  (p, a_val.v); }
  static inline void   store_u(float* p, float4 a_val) { cvex::store_u(p, a_val.v); }

  static inline float4 min  (const float4& a, const float4& b)                            { return cvex::min(a.v, b.v); }
  static inline float4 max  (const float4& a, const float4& b)                            { return cvex::max(a.v, b.v); }
  static inline float4 clamp(const float4& x, const float4& minVal, const float4& maxVal) { return cvex::clamp(x.v, minVal.v, maxVal.v); }
  static inline float4 clamp(const float4& u, float a, float b)                           { return cvex::clamp(u.v, cvex::splat(a), cvex::splat(b));  }
  static inline float4 lerp (const float4& u, const float4& v, const float t)             { return cvex::lerp(u.v, v.v, t);  }
  static inline float4 mix  (const float4& u, const float4& v, const float t)             { return cvex::lerp(u.v, v.v, t);  }
  static inline float4 mad  (const float4& a, const float4& b, const float4& c)           { return cvex::fma(a.v, b.v, c.v); }
  static inline float4 fma  (const float4& a, const float4& b, const float4& c)           { return cvex::fma(a.v, b.v, c.v); }

  static inline float4 cross(const float4& a, const float4& b) { return cvex::cross3(a.v, b.v);} 
  static inline float  dot  (const float4& a, const float4& b) { return cvex::dot4f(a.v, b.v); }
  static inline float  dot3f(const float4& a, const float4& b) { return cvex::dot3f(a.v, b.v); }
  static inline float4 dot3v(const float4& a, const float4& b) { return cvex::dot3v(a.v, b.v); }
  static inline float  dot4f(const float4& a, const float4& b) { return cvex::dot4f(a.v, b.v); }
  static inline float4 dot4v(const float4& a, const float4& b) { return cvex::dot4v(a.v, b.v); }

  static inline float  length3f(const float4& a) { return cvex::length3f(a.v); }
  static inline float  length4f(const float4& a) { return cvex::length4f(a.v); }
  static inline float4 length3v(const float4& a) { return cvex::length3v(a.v); }
  static inline float4 length4v(const float4& a) { return cvex::length4v(a.v); }
  static inline float4 normalize(const float4& u) { return u / length3f(u); }

  static inline float4 floor(const float4& a_val) { return cvex::floor(a_val.v); }
  static inline float4 ceil (const float4& a_val) { return cvex::ceil(a_val.v);  }
  static inline float4 abs (const float4& a)      { return cvex::fabs(a.v);      } 
  static inline float4 sign(const float4& a)      { return cvex::sign(a.v);      }
  static inline float4 rcp(const float4& a)       { return cvex::rcp(a.v);       }
  static inline float4 mod(float4 x, float4 y)    { return x.v - y.v * cvex::floor(x.v/y.v); }
  static inline float4 fract(float4 x)            { return x.v - cvex::floor(x.v); }
  static inline float4 sqrt(float4 x)             { return cvex::sqrt(x.v); }
  static inline float4 inversesqrt(float4 x)      { return cvex::inversesqrt(x.v); }

  static inline unsigned int color_pack_rgba(const float4 rel_col) { return cvex::color_pack_rgba(rel_col.v); }
  static inline unsigned int color_pack_bgra(const float4 rel_col) { return cvex::color_pack_bgra(rel_col.v); }

  static inline float extract_0(const float4& a_val) { return cvex::extract_0(a_val.v); }
  static inline float extract_1(const float4& a_val) { return cvex::extract_1(a_val.v); }
  static inline float extract_2(const float4& a_val) { return cvex::extract_2(a_val.v); }
  static inline float extract_3(const float4& a_val) { return cvex::extract_3(a_val.v); }

  static inline float4 splat_0(const float4& v)      { return cvex::splat_0(v.v); }
  static inline float4 splat_1(const float4& v)      { return cvex::splat_1(v.v); }
  static inline float4 splat_2(const float4& v)      { return cvex::splat_2(v.v); }
  static inline float4 splat_3(const float4& v)      { return cvex::splat_3(v.v); }  
  static inline float4 splat  (const float s)        { return cvex::splat(s); }  
  
  static inline float4 packFloatW(const float4& a, float data) { return cvex::blend(a.v, cvex::splat(data), cvex::vuint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }

  static inline float4 packIntW(const float4& a, int data)     { return cvex::blend(a.v, cvex::as_float32(cvex::splat(data)), cvex::vuint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline float4 packUIntW(const float4& a, uint data)   { return cvex::blend(a.v, cvex::as_float32(cvex::splat(data)), cvex::vuint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }

  static inline int    extractIntW(const float4& a)  { return cvex::extract_3( cvex::as_int32(a.v) );  }
  static inline uint   extractUIntW(const float4& a) { return cvex::extract_3( cvex::as_uint32(a.v) ); }
  
  static inline float hmin3(const float4 a_val) { return cvex::hmin3(a_val.v); }
  static inline float hmax3(const float4 a_val) { return cvex::hmax3(a_val.v); }

  static inline float hmin(const float4 a_val) { return cvex::hmin(a_val.v); }
  static inline float hmax(const float4 a_val) { return cvex::hmax(a_val.v); }

  static inline float4 blend(const float4 a, const float4 b, const uint4 mask) { return cvex::blend(a.v, b.v, mask.v); }
  
  static inline float4 shuffle_xzyw(float4 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline float4 shuffle_yxzw(float4 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline float4 shuffle_yzxw(float4 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline float4 shuffle_zxyw(float4 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline float4 shuffle_zyxw(float4 a_src) { return cvex::shuffle_zyxw(a_src.v); }
  static inline float4 shuffle_xyxy(float4 a_src) { return cvex::shuffle_xyxy(a_src.v); }
  static inline float4 shuffle_zwzw(float4 a_src) { return cvex::shuffle_zwzw(a_src.v); }

  static inline float4 reflect(float4 dir, float4 normal) { return cvex::reflect(dir.v, normal.v); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct float3
  {
    inline float3() : x(0), y(0), z(0) {}
    inline float3(float a, float b, float c) : x(a), y(b), z(c) {}
    inline explicit float3(float a[3]) : x(a[0]), y(a[1]), z(a[2]) {}

    inline float3(const std::initializer_list<float> a_v) { v = cvex::load_u(a_v.begin()); }
    inline float3          (cvex::vfloat4 rhs)            { v = rhs; }
    inline float3 operator=(cvex::vfloat4 rhs)            { v = rhs; return *this; }
    
    inline float& operator[](int i)       { return M[i]; } // TODO: implement via extract?
    inline float  operator[](int i) const { return M[i]; } // TODO: implement via extract?

    inline float3 operator+(const float3& b) const { return v + b.v; }
    inline float3 operator-(const float3& b) const { return v - b.v; }
    inline float3 operator*(const float3& b) const { return v * b.v; }
    inline float3 operator/(const float3& b) const { return v / b.v; }

    inline float3 operator+(const float rhs) const { return v + rhs; }
    inline float3 operator-(const float rhs) const { return v - rhs; }
    inline float3 operator*(const float rhs) const { return v * rhs; }
    inline float3 operator/(const float rhs) const { return v / rhs; }

    inline void operator*=(const float rhs) { v = v * rhs; }
    inline void operator*=(const float3& b) { v = v * b.v; }

    inline void operator/=(const float rhs) { v = v / rhs; }
    inline void operator/=(const float3& b) { v = v / b.v; }

    inline void operator+=(const float3& b) { v = v + b.v; }
    inline void operator+=(const float b  ) { v = v + b;   }

    inline void operator-=(const float3& b) { v = v - b.v; }
    inline void operator-=(const float   b) { v = v - b;   }

    inline uint4 operator> (const float3& b) const { return (v > b.v); }
    inline uint4 operator< (const float3& b) const { return (v < b.v); }
    inline uint4 operator>=(const float3& b) const { return (v >= b.v); }
    inline uint4 operator<=(const float3& b) const { return (v <= b.v); }
    inline uint4 operator==(const float3& b) const { return (v == b.v); }
    inline uint4 operator!=(const float3& b) const { return (v != b.v); }

    union
    {
      struct {float x, y, z; };
      float  M[3];
      cvex::vfloat4 v;
    };
  };

  static inline float3 operator+(const float a, const float3 b) 
  { 
    const cvex::vfloat4 res = (a + b.v);
    return float3(res); 
  }

  static inline float3 operator-(const float a, const float3 b) 
  { 
    const cvex::vfloat4 res = (a - b.v);
    return float3(res); 
  }

  static inline float3 operator*(const float a, const float3 b) 
  { 
    const cvex::vfloat4 res = (a * b.v);
    return float3(res); 
  }

  static inline float3 operator/(const float a, const float3 b) 
  { 
    const cvex::vfloat4 res = (a / b.v);
    return float3(res); 
  }

  static inline void store  (float* p, float3 a_val) { cvex::store(p, a_val.v); }
  static inline void store_u(float* p, float3 a_val) { memcpy(p, &a_val, sizeof(float)*3); }

  static inline float3 min  (const float3& a, const float3& b)                            { return cvex::min(a.v, b.v); }
  static inline float3 max  (const float3& a, const float3& b)                            { return cvex::max(a.v, b.v); }
  static inline float3 clamp(const float3& x, const float3& minVal, const float3& maxVal) { return cvex::clamp(x.v, minVal.v, maxVal.v); }
  static inline float3 clamp(const float3& u, float a, float b)                           { return float3(clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)); }
  static inline float3 lerp (const float3& u, const float3& v, const float t)             { return cvex::lerp(u.v, v.v, t); }
  static inline float3 mix  (const float3& u, const float3& v, const float t)             { return cvex::lerp(u.v, v.v, t); }

  static inline float  dot  (const float3& a, const float3& b) { return cvex::dot3f(a.v, b.v); }
  static inline float3 cross(const float3& a, const float3& b){ return cvex::cross3(a.v, b.v);} 

  static inline float  length(const float3& a) { return cvex::length3f(a.v); }
  static inline float3 normalize(const float3& u) { return u / cvex::length3f(u.v); }

  static inline float3 floor(const float3& a_val) { return cvex::floor(a_val.v); }
  static inline float3 ceil (const float3& a_val) { return cvex::ceil(a_val.v);  }
  static inline float3 abs (const float3& a)      { return cvex::fabs(a.v);      } 
  static inline float3 sign(const float3& a)      { return cvex::sign(a.v);      }
  static inline float3 rcp(const float3& a)       { return cvex::rcp(a.v);       }
  static inline float3 mod(float3 x, float3 y)    { return x.v - y.v * cvex::floor(x.v/y.v); }
  static inline float3 fract(float3 x)            { return x.v - cvex::floor(x.v); }
  static inline float3 sqrt(float3 x)             { return cvex::sqrt(x.v); }
  static inline float3 inversesqrt(float3 x)      { return cvex::inversesqrt(x.v); }

  static inline unsigned int color_pack_rgba(const float3 rel_col) { return cvex::color_pack_rgba(rel_col.v); }
  static inline unsigned int color_pack_bgra(const float3 rel_col) { return cvex::color_pack_bgra(rel_col.v); }

  static inline float extract_0(const float3& a_val) { return cvex::extract_0(a_val.v); }
  static inline float extract_1(const float3& a_val) { return cvex::extract_1(a_val.v); }
  static inline float extract_2(const float3& a_val) { return cvex::extract_2(a_val.v); }
  static inline float extract_3(const float3& a_val) { return cvex::extract_3(a_val.v); }

  static inline float3 splat_0(const float3& v)      { return cvex::splat_0(v.v); }
  static inline float3 splat_1(const float3& v)      { return cvex::splat_1(v.v); }
  static inline float3 splat_2(const float3& v)      { return cvex::splat_2(v.v); }  
 
  static inline float hmin(const float3 a_val) { return cvex::hmin3(a_val.v); }
  static inline float hmax(const float3 a_val) { return cvex::hmax3(a_val.v); }

  static inline float3 blend(const float3 a, const float3 b, const uint4 mask) { return cvex::blend(a.v, b.v, mask.v); }
  
  static inline float3 shuffle_xzy(float3 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline float3 shuffle_yxz(float3 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline float3 shuffle_yzx(float3 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline float3 shuffle_zxy(float3 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline float3 shuffle_zyx(float3 a_src) { return cvex::shuffle_zyxw(a_src.v); }
  
  static inline float3 reflect(float3 dir, float3 normal) { return cvex::reflect(dir.v, normal.v); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  

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

  /**
  \brief this class use colmajor memory layout for effitient vector-matrix operations
  */
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

    inline float4x4 operator*(const float4x4& rhs)
    {
      // transpose will change multiplication order (due to in fact we use column major)
      //
      float4x4 res;
      cvex::mat4_rowmajor_mul_mat4((float*)res.m_col, (const float*)rhs.m_col, (const float*)m_col); 
      return res;
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

    struct RowTmp 
    {
      float4x4* self;
      int       row;
      inline float& operator[](int col)       { return self->m_col[col][row]; }
      inline float  operator[](int col) const { return self->m_col[col][row]; }
    };

    inline RowTmp operator[](int a_row) 
    {
      RowTmp row;
      row.self = this;
      row.row  = a_row;
      return row;
    }

  //private:
    float4 m_col[4];
  };

  static inline float4 operator*(const float4x4& m, const float4& v)
  {
    float4 res;
    cvex::mat4_colmajor_mul_vec4((float*)&res, (const float*)&m, (const float*)&v);
    return res;
  }

  static inline float4x4 transpose(const float4x4& rhs)
  {
    float4x4 res;
    cvex::transpose4((const cvex::vfloat4*)&rhs, (cvex::vfloat4*)&res);
    return res;
  }

  static inline float4x4 translate4x4(float3 t)
  {
    float4x4 res;
    res.set_col(3, float4{t.x,  t.y,  t.z, 1.0f });
    return res;
  }

  static inline float4x4 scale4x4(float3 t)
  {
    float4x4 res;
    res.set_col(0, float4{t.x, 0.0f, 0.0f,  0.0f});
    res.set_col(1, float4{0.0f, t.y, 0.0f,  0.0f});
    res.set_col(2, float4{0.0f, 0.0f,  t.z, 0.0f});
    res.set_col(3, float4{0.0f, 0.0f, 0.0f, 1.0f});
    return res;
  }

  static inline float4x4 rotate4x4X(float phi)
  {
    float4x4 res;
    res.set_col(0, float4{1.0f,      0.0f,       0.0f, 0.0f  });
    res.set_col(1, float4{0.0f, +cosf(phi),  +sinf(phi), 0.0f});
    res.set_col(2, float4{0.0f, -sinf(phi),  +cosf(phi), 0.0f});
    res.set_col(3, float4{0.0f,      0.0f,       0.0f, 1.0f  });
    return res;
  }

  static inline float4x4 rotate4x4Y(float phi)
  {
    float4x4 res;
    res.set_col(0, float4{+cosf(phi), 0.0f, -sinf(phi), 0.0f});
    res.set_col(1, float4{     0.0f, 1.0f,      0.0f, 0.0f  });
    res.set_col(2, float4{+sinf(phi), 0.0f, +cosf(phi), 0.0f});
    res.set_col(3, float4{     0.0f, 0.0f,      0.0f, 1.0f  });
    return res;
  }

  static inline float4x4 rotate4x4Z(float phi)
  {
    float4x4 res;
    res.set_col(0, float4{+cosf(phi), sinf(phi), 0.0f, 0.0f});
    res.set_col(1, float4{-sinf(phi), cosf(phi), 0.0f, 0.0f});
    res.set_col(2, float4{     0.0f,     0.0f, 1.0f, 0.0f  });
    res.set_col(3, float4{     0.0f,     0.0f, 0.0f, 1.0f  });
    return res;
  }

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
    CVEX_ALIGNED(16) float tmp[12]; // temp array for pairs
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
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  //static inline int max  (int a, int b)        { return a > b ? a : b; }                                    
  //static inline int min  (int a, int b)        { return a < b ? a : b; }                                    
  //static inline int clamp(int u, int a, int b) { const int   r = (a > u) ? a : u; return (r < b) ? r : b; } 

  inline float rnd(float s, float e)
  {
    const float t = (float)(rand()) / (float)RAND_MAX;
    return s + t*(e - s);
  }

  template<typename T> inline T SQR(T x) { return x * x; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static inline float  lengthSquare(const float3& u) { return dot(u,u); }
  static inline float  maxcomp     (const float3& u) { return fmax(u.x, fmax(u.y, u.z)); }
  static inline float  mincomp     (const float3& u) { return fmin(u.x, fmin(u.y, u.z)); }

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
  static inline float  lerp(float u, float v, float t) { return u + t * (v - u);  } 
  static inline float  mix (float u, float v, float t) { return u + t * (v - u);  } 

  static inline float smoothstep(float edge0, float edge1, float x)
  {
    float  tVal = (x - edge0) / (edge1 - edge0);
    float  t    = fmin(fmax(tVal, 0.0f), 1.0f); 
    return t * t * (3.0f - 2.0f * t);
  }

  static inline float3 operator*(const float4x4& m, const float3& v)
  {
    float4 v2 = float4{v.x, v.y, v.z, 1.0f}; //cvex::to_float4(v, 1.0f);
    float3 res;                              // yes we know sizeof(float3) == sizeof(float4)
    cvex::mat4_colmajor_mul_vec4((float*)&res, (const float*)&m, (const float*)&v2);
    return res;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
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

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct uint3
  {
    uint3() :x(0), y(0), z(0) {}
    uint3(unsigned int a, unsigned int b, unsigned int c) : x(a), y(b), z(c) {}

    unsigned int x, y, z;
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


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


  static inline float4 make_float4(float a, float b, float c, float d) { return float4{a,b,c,d}; }
  static inline float3 make_float3(float a, float b, float c)          { return float3{a,b,c};   }
  static inline float3 to_float3(float4 f4)                            { return float3(f4.v); }
  static inline float4 to_float4(float3 v, float w) { return cvex::blend(v.v, cvex::splat(w), cvex::vuint4{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 }); }

  static inline int4   to_int32  (const float4& a) { return cvex::to_int32(a.v);  }
  static inline int4   to_int32  (const uint4& a)  { return cvex::to_int32(a.v); }
  static inline uint4  to_uint32 (const float4& a) { return cvex::to_uint32(a.v); }
  static inline uint4  to_uint32 (const int4& a)   { return cvex::to_uint32(a.v); }

  static inline float4 to_float32(const  int4& a)  { return cvex::to_float32(a.v); }
  static inline float4 to_float32(const uint4& a)  { return cvex::to_float32(a.v); }

  static inline float4 as_float32(const int4 a_val)   { return cvex::as_float32(a_val.v); }
  static inline float4 as_float32(const uint4 a_val)  { return cvex::as_float32(a_val.v); }
  static inline int4   as_int32  (const float4 a_val) { return cvex::as_int32  (a_val.v); }
  static inline uint4  as_uint32 (const float4 a_val) { return cvex::as_uint32 (a_val.v); }

  //static inline int3   to_int32  (const float3& a) { return cvex::to_int32(a.v); }
  //static inline uint3  to_uint32 (const float3& a) { return cvex::to_uint32(a.v); }
  //static inline float3 to_float32(const  int3& a)  { return cvex::to_float32(a.v); }
  //static inline float3 to_float32(const uint3& a)  { return cvex::to_float32(a); }
  
  //static inline float3 as_float32(const int3 a_val)   { return cvex::as_float32(a_val.v); }
  //static inline float3 as_float32(const uint3 a_val)  { return cvex::as_float32(a_val); }
  //static inline int3   as_int32  (const float3 a_val) { return cvex::as_int32  (a_val.v); }
  //static inline uint3  as_uint32 (const float3 a_val) { return cvex::as_uint32 (a_val.v); }

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

};

#endif
