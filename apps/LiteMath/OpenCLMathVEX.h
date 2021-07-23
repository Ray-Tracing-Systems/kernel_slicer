#ifndef VFLOAT4_ALL_H
#define VFLOAT4_ALL_H

//#include "vfloat4_gcc.h"
//#include "vfloat4_x64.h"
//#include "vfloat4_arm.h"
// TODO: __riscv__
// TODO: __mips__
// TODO: __ppc__ 

#ifdef WIN32
  #include "vfloat4_x64.h"
#else
  #if defined(__arm__) or defined(__aarch64__)
    #include "vfloat4_arm.h"
  #elif defined(__clang__)
    #include "vfloat4_x64.h"
  #elif defined(__GNUC__)
    #include "vfloat4_gcc.h"
  #else
    #include "vfloat4_x64.h"
  #endif  
#endif

#include <cmath>
#include <initializer_list>
#include <limits>

#include <cstring> // for memcpy

#ifdef min
#undef min
#endif

#ifdef max
#undef min
#endif

namespace LiteMath
{ 
  typedef unsigned int uint;

  constexpr float EPSILON      = 1e-6f;
  constexpr float DEG_TO_RAD   = float(3.14159265358979323846f) / 180.0f;
  constexpr float INF_POSITIVE = +std::numeric_limits<float>::infinity();
  constexpr float INF_NEGATIVE = -std::numeric_limits<float>::infinity();

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
  static inline uint  clamp(uint u,  uint a,  uint b)  { return std::min(std::max(a, u), b); }
  static inline int   clamp(int u,   int a,   int b)   { return std::min(std::max(a, u), b); }

  inline float rnd(float s, float e)
  {
    const float t = (float)(rand()) / (float)RAND_MAX;
    return s + t*(e - s);
  }
  
  template<typename T> inline T SQR(T x) { return x * x; }

  static inline float  lerp(float u, float v, float t) { return u + t * (v - u);  } 
  static inline float  mix (float u, float v, float t) { return u + t * (v - u);  } 

  static inline float smoothstep(float edge0, float edge1, float x)
  {
    float  tVal = (x - edge0) / (edge1 - edge0);
    float  t    = fmin(fmax(tVal, 0.0f), 1.0f); 
    return t * t * (3.0f - 2.0f * t);
  }

  using std::min;
  using std::max;
  using std::sqrt;
  using std::abs;

  static inline float fract(float x)        { return x - std::floor(x); }
  static inline float mod(float x, float y) { return x - y * std::floor(x/y); }
  static inline float sign(float x) // TODO: on some architectures we can try to effitiently check sign bit       
  { 
    if(x == 0.0f)     return 0.0f;
    else if(x < 0.0f) return -1.0f;
    else              return +1.0f;
  } 
  
  static inline float inversesqrt(float x) { return 1.0f/std::sqrt(x); }
  static inline float rcp(float x)         { return 1.0f/x; }

  static inline int sign(int x) // TODO: on some architectures we can try to effitiently check sign bit       
  { 
    if(x == 0)     return 0;
    else if(x < 0) return -1;
    else           return +1;
  } 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct CVEX_ALIGNED(16) uint4
  {
    inline uint4()                               : x(0), y(0), z(0), w(0) {}
    inline uint4(uint a, uint b, uint c, uint d) : x(a), y(b), z(c), w(d) {}
    inline explicit uint4(uint a[4])             : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

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

  static inline uint4 operator> (const uint4 a, const uint4 b) { return uint4{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0, a.w >  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator< (const uint4 a, const uint4 b) { return uint4{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0, a.w <  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator>=(const uint4 a, const uint4 b) { return uint4{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0, a.w >= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator<=(const uint4 a, const uint4 b) { return uint4{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0, a.w <= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator==(const uint4 a, const uint4 b) { return uint4{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0, a.w == b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator!=(const uint4 a, const uint4 b) { return uint4{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0, a.w != b.w ? 0xFFFFFFFF : 0}; }

  static inline uint4 operator&(const uint4 a, const uint4 b) { return uint4(a.v & b.v); }
  static inline uint4 operator|(const uint4 a, const uint4 b) { return uint4(a.v | b.v); }
  static inline uint4 operator~(const uint4 a)                { return uint4(~a.v); }
  static inline uint4 operator>>(const uint4 a, uint b)       { return uint4{a.x >> b, a.y >> b, a.z >> b, a.w >> b}; }
  static inline uint4 operator<<(const uint4 a, uint b)       { return uint4{a.x << b, a.y << b, a.z << b, a.w << b}; }
  
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
  static inline uint  hmin (const uint4 a_val) { return cvex::hmin(a_val.v); }
  static inline uint  hmax (const uint4 a_val) { return cvex::hmax(a_val.v); }
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct CVEX_ALIGNED(16) uint3
  {
    inline uint3()                       { }
    inline uint3(uint a, uint b, uint c) { CVEX_ALIGNED(16) uint data[4] = {a,b,c,1};          v = cvex::load(data); }
    inline explicit uint3(uint a[3])     { CVEX_ALIGNED(16) uint data[4] = {a[0],a[1],a[2],1}; v = cvex::load(data); }

    inline uint3(const std::initializer_list<uint> a_v) { v = cvex::load_u(a_v.begin()); }
    inline uint3(cvex::vuint4 rhs) { v = rhs; }
    inline uint3 operator=(cvex::vuint4 rhs) { v = rhs; return *this; }

    inline uint& operator[](int i)       { return M[i]; }
    inline uint  operator[](int i) const { return M[i]; }

    inline uint3 operator+(const uint3& b) const { return v + b.v; }
    inline uint3 operator-(const uint3& b) const { return v - b.v; }
    inline uint3 operator*(const uint3& b) const { return v * b.v; }
    inline uint3 operator/(const uint3& b) const { return v / b.v; }

    inline uint3 operator+(const uint rhs) const { return v + rhs; }
    inline uint3 operator-(const uint rhs) const { return v - rhs; }
    inline uint3 operator*(const uint rhs) const { return v * rhs; }
    inline uint3 operator/(const uint rhs) const { return v / rhs; }

    inline void operator*=(const uint rhs) { v = v * rhs; }
    inline void operator*=(const uint3& b) { v = v * b.v; }
    inline void operator/=(const uint rhs) { v = v / rhs; }
    inline void operator/=(const uint3& b) { v = v / b.v; }
    inline void operator+=(const uint b  ) { v = v + b;   }
    inline void operator+=(const uint3& b) { v = v + b.v; }
    inline void operator-=(const uint   b) { v = v - b;   }
    inline void operator-=(const uint3& b) { v = v - b.v; }

    union
    {
      struct { uint x, y, z; };
      uint M[3];
      cvex::vuint4 v;
    };
  };

  static inline uint3 operator+(const uint a, const uint3 b) { return uint3(a + b.v); }
  static inline uint3 operator-(const uint a, const uint3 b) { return uint3(a - b.v); }
  static inline uint3 operator*(const uint a, const uint3 b) { return uint3(a * b.v); }
  static inline uint3 operator/(const uint a, const uint3 b) { return uint3(a / b.v); }

  static inline void  store  (uint* p, uint3 a_val) { cvex::store(p, a_val.v); }
  static inline void  store_u(uint* p, uint3 a_val) { memcpy(p, &a_val, sizeof(uint)*3); }

  static inline uint3 operator> (const uint3 a, const uint3 b) { return uint3(a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0); }
  static inline uint3 operator< (const uint3 a, const uint3 b) { return uint3(a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0); }
  static inline uint3 operator>=(const uint3 a, const uint3 b) { return uint3(a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0); }
  static inline uint3 operator<=(const uint3 a, const uint3 b) { return uint3(a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0); }
  static inline uint3 operator==(const uint3 a, const uint3 b) { return uint3(a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0); }
  static inline uint3 operator!=(const uint3 a, const uint3 b) { return uint3(a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0); }

  static inline uint3 operator& (const uint3 a, const uint3 b) { return uint3(a.v & b.v); }
  static inline uint3 operator| (const uint3 a, const uint3 b) { return uint3(a.v | b.v); }
  static inline uint3 operator~ (const uint3 a)                { return uint3(~a.v); }
  static inline uint3 operator>>(const uint3 a, uint b)        { return uint3(a.x >> b, a.y >> b, a.z >> b); }
  static inline uint3 operator<<(const uint3 a, uint b)        { return uint3(a.x << b, a.y << b, a.z << b); }
  
  static inline uint3 min  (const uint3& a,   const uint3& b) { return cvex::min(a.v, b.v); }
  static inline uint3 max  (const uint3& a,   const uint3& b) { return cvex::max(a.v, b.v); }
  static inline uint3 clamp(const uint3& a_x, const uint3& a_min, const uint3& a_max) { return cvex::clamp(a_x.v, a_min.v, a_max.v); }
  static inline uint3 clamp(const uint3& u, uint a, uint b)                           { return cvex::clamp(u.v, cvex::splat(a), cvex::splat(b)); }

  static inline bool any_of (const uint3 a)  { return cvex::any_of(a.v); }
  static inline bool all_of (const uint3 a)  { return cvex::all_of(a.v); }

  static inline uint3 blend(const uint3 a, const uint3 b, const uint3 mask) { return cvex::blend(a.v, b.v, mask.v); }

  static inline uint3 shuffle_xzy(uint3 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline uint3 shuffle_yxz(uint3 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline uint3 shuffle_yzx(uint3 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline uint3 shuffle_zxy(uint3 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline uint3 shuffle_zyx(uint3 a_src) { return cvex::shuffle_zyxw(a_src.v); }

  static inline uint extract_0(const uint3& a_val) { return cvex::extract_0(a_val.v); }
  static inline uint extract_1(const uint3& a_val) { return cvex::extract_1(a_val.v); }
  static inline uint extract_2(const uint3& a_val) { return cvex::extract_2(a_val.v); }

  static inline uint3 splat_0(const uint3& v)   { return cvex::splat_0(v.v); }
  static inline uint3 splat_1(const uint3& v)   { return cvex::splat_1(v.v); }
  static inline uint3 splat_2(const uint3& v)   { return cvex::splat_2(v.v); }  

  static inline uint  hmin(const uint3 a_val) { return cvex::hmin3(a_val.v); } 
  static inline uint  hmax(const uint3 a_val) { return cvex::hmax3(a_val.v); } 
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct uint2
  {
    inline uint2() : x(0), y(0) {}
    inline uint2(unsigned int a, unsigned int b) : x(a), y(b) {}
    inline explicit uint2(uint a[2]) : x(a[0]), y(a[1]) {}
    
    inline uint& operator[](int i)       { return M[i]; }
    inline uint  operator[](int i) const { return M[i]; }

    union
    {
      struct {uint x, y; };
      uint M[2];
    };
  };

  static inline uint2 operator * (const uint2& u, uint v) { return uint2{u.x * v, u.y * v}; }
  static inline uint2 operator / (const uint2& u, uint v) { return uint2{u.x / v, u.y / v}; }
  static inline uint2 operator * (uint v, const uint2& u) { return uint2{v * u.x, v * u.y}; }
  static inline uint2 operator / (uint v, const uint2& u) { return uint2{v / u.x, v / u.y}; }

  static inline uint2 operator + (const uint2& u, uint v) { return uint2{u.x + v, u.y + v}; }
  static inline uint2 operator - (const uint2& u, uint v) { return uint2{u.x - v, u.y - v}; }
  static inline uint2 operator + (uint v, const uint2& u) { return uint2{v + u.x, v + u.y}; }
  static inline uint2 operator - (uint v, const uint2& u) { return uint2{v - u.x, v - u.y}; }

  static inline uint2 operator + (const uint2& u, const uint2& v) { return uint2{u.x + v.x, u.y + v.y}; }
  static inline uint2 operator - (const uint2& u, const uint2& v) { return uint2{u.x - v.x, u.y - v.y}; }
  static inline uint2 operator * (const uint2& u, const uint2& v) { return uint2{u.x * v.x, u.y * v.y}; }
  static inline uint2 operator / (const uint2& u, const uint2& v) { return uint2{u.x / v.x, u.y / v.y}; }

  static inline uint2& operator += (uint2& u, const uint2& v) { u.x += v.x; u.y += v.y; return u; }
  static inline uint2& operator -= (uint2& u, const uint2& v) { u.x -= v.x; u.y -= v.y; return u; }
  static inline uint2& operator *= (uint2& u, const uint2& v) { u.x *= v.x; u.y *= v.y; return u; }
  static inline uint2& operator /= (uint2& u, const uint2& v) { u.x /= v.x; u.y /= v.y; return u; }

  static inline uint2& operator += (uint2& u, uint v) { u.x += v; u.y += v; return u; }
  static inline uint2& operator -= (uint2& u, uint v) { u.x -= v; u.y -= v; return u; }
  static inline uint2& operator *= (uint2& u, uint v) { u.x *= v; u.y *= v; return u; }
  static inline uint2& operator /= (uint2& u, uint v) { u.x /= v; u.y /= v; return u; }
  
  static inline uint2 operator> (const uint2& a, const uint2& b) { return uint2{a[0] > b[0]  ? 0xFFFFFFFF : 0, a[1] > b[1]  ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator< (const uint2& a, const uint2& b) { return uint2{a[0] < b[0]  ? 0xFFFFFFFF : 0, a[1] < b[1]  ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator>=(const uint2& a, const uint2& b) { return uint2{a[0] >= b[0] ? 0xFFFFFFFF : 0, a[1] >= b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator<=(const uint2& a, const uint2& b) { return uint2{a[0] <= b[0] ? 0xFFFFFFFF : 0, a[1] <= b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator==(const uint2& a, const uint2& b) { return uint2{a[0] == b[0] ? 0xFFFFFFFF : 0, a[1] == b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator!=(const uint2& a, const uint2& b) { return uint2{a[0] != b[0] ? 0xFFFFFFFF : 0, a[1] != b[1] ? 0xFFFFFFFF : 0}; }
  
  static inline uint2 operator& (const uint2 a, const uint2 b) { return uint2{a.x & b.x, a.y & b.y}; }
  static inline uint2 operator| (const uint2 a, const uint2 b) { return uint2{a.x | b.x, a.y | b.y}; }
  static inline uint2 operator~ (const uint2 a)                { return uint2{~a.x     , ~a.y     }; }
  static inline uint2 operator>>(const uint2 a, const uint b)  { return uint2{a.x >> b , a.y >> b }; }
  static inline uint2 operator<<(const uint2 a, const uint b)  { return uint2{a.x << b , a.y << b }; }

  static inline void store  (uint* p, uint2 a_val) { memcpy(p, &a_val, sizeof(uint)*2); }
  static inline void store_u(uint* p, uint2 a_val) { memcpy(p, &a_val, sizeof(uint)*2); }  
  
  static inline uint dot(const uint2& u, const uint2& v)  { return (u.x*v.x + u.y*v.y); }
  
  static inline uint2 min  (const uint2& a, const uint2& b)   { return uint2{ std::min(a[0], b[0]), std::min(a[1], b[1])}; }
  static inline uint2 max  (const uint2& a, const uint2& b)   { return uint2{ std::max(a[0], b[0]), std::max(a[1], b[1])}; }
  static inline uint2 clamp(const uint2& u, uint a, uint b)  { return uint2{ clamp(u.x, a, b),     clamp(u.y, a, b)};     }
  
  //static inline uint2 abs (const uint2& a)     { return uint2{std::abs(a.M[0]),   std::abs(a.M[1])};   } 
  //static inline uint2 sign(const uint2& a)     { return uint2{sign(a.M[0]),       sign(a.M[1])};       }
  
  static inline uint extract_0(const uint2& a) { return a.M[0]; }
  static inline uint extract_1(const uint2& a) { return a.M[1]; }

  static inline uint2 splat_0(const uint2& a)  { return uint2{ a.M[0], a.M[0] }; }
  static inline uint2 splat_1(const uint2& a)  { return uint2{ a.M[1], a.M[1] }; }

  static inline uint hmin    (const uint2& a) { return std::min(a.M[0], a.M[1]); }
  static inline uint hmax    (const uint2& a) { return std::max(a.M[0], a.M[1]); }
 
  static inline uint2 blend(const uint2 a, const uint2 b, const uint2 mask) { return uint2{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y}; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct CVEX_ALIGNED(16) int4
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

    inline void operator*=(const int rhs) { v = v * rhs; }
    inline void operator*=(const int4& b) { v = v * b.v; }
    inline void operator/=(const int rhs) { v = v / rhs; }
    inline void operator/=(const int4& b) { v = v / b.v; }
    inline void operator+=(const int b  ) { v = v + b;   }
    inline void operator+=(const int4& b) { v = v + b.v; }
    inline void operator-=(const int   b) { v = v - b;   }
    inline void operator-=(const int4& b) { v = v - b.v; }

    inline uint4 operator> (const int4& b) const { return (v > b.v); }
    inline uint4 operator< (const int4& b) const { return (v < b.v); }
    inline uint4 operator>=(const int4& b) const { return (v >= b.v); }
    inline uint4 operator<=(const int4& b) const { return (v <= b.v); }
    inline uint4 operator==(const int4& b) const { return (v == b.v); }
    inline uint4 operator!=(const int4& b) const { return (v != b.v); }

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
  static inline int4 abs (const int4 a)                                           { return int4{std::abs(a.x), std::abs(a.y), std::abs(a.z), std::abs(a.w)}; } 
  static inline int4 sign(const int4 a)                                           { return int4{sign(a.x), sign(a.y), sign(a.z), sign(a.w)}; }
  

  static inline bool any_of (const int4 a) { return cvex::any_of(a.v); }
  static inline bool all_of (const int4 a) { return cvex::all_of(a.v); }

  static inline int4 blend(const int4 a, const int4 b, const uint4 mask) { return cvex::blend(a.v, b.v, mask.v); }

  static inline int4 shuffle_xzyw(int4 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline int4 shuffle_yxzw(int4 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline int4 shuffle_yzxw(int4 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline int4 shuffle_zxyw(int4 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline int4 shuffle_zyxw(int4 a_src) { return cvex::shuffle_zyxw(a_src.v); }
  static inline int4 shuffle_xyxy(int4 a_src) { return cvex::shuffle_xyxy(a_src.v); }
  static inline int4 shuffle_zwzw(int4 a_src) { return cvex::shuffle_zwzw(a_src.v); }

  static inline int extract_0(const int4& a_val) { return cvex::extract_0(a_val.v); }
  static inline int extract_1(const int4& a_val) { return cvex::extract_1(a_val.v); }
  static inline int extract_2(const int4& a_val) { return cvex::extract_2(a_val.v); }
  static inline int extract_3(const int4& a_val) { return cvex::extract_3(a_val.v); }

  static inline int4 splat_0(const int4& v)   { return cvex::splat_0(v.v); }
  static inline int4 splat_1(const int4& v)   { return cvex::splat_1(v.v); }
  static inline int4 splat_2(const int4& v)   { return cvex::splat_2(v.v); }
  static inline int4 splat_3(const int4& v)   { return cvex::splat_3(v.v); }  
  static inline int4 splat  (const int s)     { return cvex::splat(s); }  

  static inline int  hmin3(const int4 a_val) { return cvex::hmin3(a_val.v); }
  static inline int  hmax3(const int4 a_val) { return cvex::hmax3(a_val.v); } 
  static inline int  hmin (const int4 a_val) { return cvex::hmin(a_val.v); }
  static inline int  hmax (const int4 a_val) { return cvex::hmax(a_val.v); }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct CVEX_ALIGNED(16) int3
  {
    inline int3() { }
    inline int3(int a, int b, int c) { CVEX_ALIGNED(16) int data[4] = {a,b,c,1};          v = cvex::load(data); }
    inline explicit int3(int a[3])   { CVEX_ALIGNED(16) int data[4] = {a[0],a[1],a[2],1}; v = cvex::load(data); }

    inline int3(const std::initializer_list<int> a_v) { v = cvex::load_u(a_v.begin()); }
    inline int3(cvex::vint4 rhs) { v = rhs; }
    inline int3 operator=(cvex::vint4 rhs) { v = rhs; return *this; }

    inline int& operator[](int i)       { return M[i]; }
    inline int  operator[](int i) const { return M[i]; }

    inline int3 operator+(const int3& b) const { return v + b.v; }
    inline int3 operator-(const int3& b) const { return v - b.v; }
    inline int3 operator*(const int3& b) const { return v * b.v; }
    inline int3 operator/(const int3& b) const { return v / b.v; }

    inline int3 operator+(const int rhs) const { return v + rhs; }
    inline int3 operator-(const int rhs) const { return v - rhs; }
    inline int3 operator*(const int rhs) const { return v * rhs; }
    inline int3 operator/(const int rhs) const { return v / rhs; }

    inline void operator*=(const int rhs) { v = v * rhs; }
    inline void operator*=(const int3& b) { v = v * b.v; }
    inline void operator/=(const int rhs) { v = v / rhs; }
    inline void operator/=(const int3& b) { v = v / b.v; }
    inline void operator+=(const int b  ) { v = v + b;   }
    inline void operator+=(const int3& b) { v = v + b.v; }
    inline void operator-=(const int   b) { v = v - b;   }
    inline void operator-=(const int3& b) { v = v - b.v; }

    inline uint3 operator> (const int3& b) const { return (v > b.v); }
    inline uint3 operator< (const int3& b) const { return (v < b.v); }
    inline uint3 operator>=(const int3& b) const { return (v >= b.v); }
    inline uint3 operator<=(const int3& b) const { return (v <= b.v); }
    inline uint3 operator==(const int3& b) const { return (v == b.v); }
    inline uint3 operator!=(const int3& b) const { return (v != b.v); }

    union
    {
      struct { int x, y, z; };
      int M[3];
      cvex::vint4 v;
    };
  };

  static inline int3 operator+(const int a, const int3 b) { return int3(a + b.v); }
  static inline int3 operator-(const int a, const int3 b) { return int3(a - b.v); }
  static inline int3 operator*(const int a, const int3 b) { return int3(a * b.v); }
  static inline int3 operator/(const int a, const int3 b) { return int3(a / b.v); }

  static inline void  store  (int* p, int3 a_val) { cvex::store(p, a_val.v); }
  static inline void  store_u(int* p, int3 a_val) { memcpy(p, &a_val, sizeof(int)*3); }

  static inline int3 operator& (const int3 a, const int3 b) { return int3(a.v & b.v); }
  static inline int3 operator| (const int3 a, const int3 b) { return int3(a.v | b.v); }
  static inline int3 operator~ (const int3 a)               { return int3(~a.v); }
  static inline int3 operator>>(const int3 a, const int b)  { return int3(a.v >> b); }
  static inline int3 operator<<(const int3 a, const int b)  { return int3(a.v << b); }
  
  static inline int3 min  (const int3 a,   const int3 b) { return cvex::min(a.v, b.v); }
  static inline int3 max  (const int3 a,   const int3 b) { return cvex::max(a.v, b.v); }
  static inline int3 clamp(const int3 a_x, const int3 a_min, const int3& a_max) { return cvex::clamp(a_x.v, a_min.v, a_max.v); }
  static inline int3 clamp(const int3 u, int a, int b)                          { return cvex::clamp(u.v, cvex::splat(a), cvex::splat(b)); }
  static inline int3 abs  (const int3 a)                                        { return int3(std::abs(a.x), std::abs(a.y), std::abs(a.z)); } 
  static inline int3 sign (const int3 a)                                        { return int3(sign(a.x), sign(a.y), sign(a.z)); }
  
  static inline int3 blend(const int3 a, const int3 b, const uint3 mask) { return cvex::blend(a.v, b.v, mask.v); }

  static inline int3 shuffle_xzy(int3 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline int3 shuffle_yxz(int3 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline int3 shuffle_yzx(int3 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline int3 shuffle_zxy(int3 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline int3 shuffle_zyx(int3 a_src) { return cvex::shuffle_zyxw(a_src.v); }

  static inline int extract_0(const int3& a_val) { return cvex::extract_0(a_val.v); }
  static inline int extract_1(const int3& a_val) { return cvex::extract_1(a_val.v); }
  static inline int extract_2(const int3& a_val) { return cvex::extract_2(a_val.v); }

  static inline int3 splat_0(const int3& v)   { return cvex::splat_0(v.v); }
  static inline int3 splat_1(const int3& v)   { return cvex::splat_1(v.v); }
  static inline int3 splat_2(const int3& v)   { return cvex::splat_2(v.v); }  

  static inline int  hmin(const int3 a_val) { return cvex::hmin3(a_val.v); } 
  static inline int  hmax(const int3 a_val) { return cvex::hmax3(a_val.v); } 
  
  struct int2
  {
    inline int2() : x(0), y(0) {}
    inline int2(int a, int b) : x(a), y(b) {}
    inline explicit int2(uint a[2]) : x(a[0]), y(a[1]) {}
    
    inline int& operator[](int i)       { return M[i]; }
    inline int  operator[](int i) const { return M[i]; }

    union
    {
      struct {int x, y; };
      int M[2];
    };
  };

  static inline int2 operator * (const int2& u, int v) { return int2{u.x * v, u.y * v}; }
  static inline int2 operator / (const int2& u, int v) { return int2{u.x / v, u.y / v}; }
  static inline int2 operator * (int v, const int2& u) { return int2{v * u.x, v * u.y}; }
  static inline int2 operator / (int v, const int2& u) { return int2{v / u.x, v / u.y}; }

  static inline int2 operator + (const int2& u, int v) { return int2{u.x + v, u.y + v}; }
  static inline int2 operator - (const int2& u, int v) { return int2{u.x - v, u.y - v}; }
  static inline int2 operator + (int v, const int2& u) { return int2{v + u.x, v + u.y}; }
  static inline int2 operator - (int v, const int2& u) { return int2{v - u.x, v - u.y}; }

  static inline int2 operator + (const int2& u, const int2& v) { return int2{u.x + v.x, u.y + v.y}; }
  static inline int2 operator - (const int2& u, const int2& v) { return int2{u.x - v.x, u.y - v.y}; }
  static inline int2 operator * (const int2& u, const int2& v) { return int2{u.x * v.x, u.y * v.y}; }
  static inline int2 operator / (const int2& u, const int2& v) { return int2{u.x / v.x, u.y / v.y}; }
  static inline int2 operator - (const int2& v) { return {-v.x, -v.y}; }

  static inline int2& operator += (int2& u, const int2& v) { u.x += v.x; u.y += v.y; return u; }
  static inline int2& operator -= (int2& u, const int2& v) { u.x -= v.x; u.y -= v.y; return u; }
  static inline int2& operator *= (int2& u, const int2& v) { u.x *= v.x; u.y *= v.y; return u; }
  static inline int2& operator /= (int2& u, const int2& v) { u.x /= v.x; u.y /= v.y; return u; }

  static inline int2& operator += (int2& u, int v) { u.x += v; u.y += v; return u; }
  static inline int2& operator -= (int2& u, int v) { u.x -= v; u.y -= v; return u; }
  static inline int2& operator *= (int2& u, int v) { u.x *= v; u.y *= v; return u; }
  static inline int2& operator /= (int2& u, int v) { u.x /= v; u.y /= v; return u; }
  
  static inline uint2 operator> (const int2& a, const int2& b) { return uint2{a[0] > b[0]  ? 0xFFFFFFFF : 0, a[1] > b[1]  ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator< (const int2& a, const int2& b) { return uint2{a[0] < b[0]  ? 0xFFFFFFFF : 0, a[1] < b[1]  ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator>=(const int2& a, const int2& b) { return uint2{a[0] >= b[0] ? 0xFFFFFFFF : 0, a[1] >= b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator<=(const int2& a, const int2& b) { return uint2{a[0] <= b[0] ? 0xFFFFFFFF : 0, a[1] <= b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator==(const int2& a, const int2& b) { return uint2{a[0] == b[0] ? 0xFFFFFFFF : 0, a[1] == b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator!=(const int2& a, const int2& b) { return uint2{a[0] != b[0] ? 0xFFFFFFFF : 0, a[1] != b[1] ? 0xFFFFFFFF : 0}; }
  
  static inline int2 operator& (const int2 a, const int2 b) { return int2{a.x & b.x, a.y & b.y}; }
  static inline int2 operator| (const int2 a, const int2 b) { return int2{a.x | b.x, a.y | b.y}; }
  static inline int2 operator~ (const int2 a)               { return int2{~a.x     , ~a.y     }; }
  static inline int2 operator>>(const int2 a, const int b)  { return int2{a.x >> b , a.y >> b }; }
  static inline int2 operator<<(const int2 a, const int b)  { return int2{a.x << b , a.y << b }; }

  static inline void store  (int* p, int2 a_val) { memcpy(p, &a_val, sizeof(int)*2); }
  static inline void store_u(int* p, int2 a_val) { memcpy(p, &a_val, sizeof(int)*2); }  
  
  static inline int dot(const int2& u, const int2& v)  { return (u.x*v.x + u.y*v.y); }
  
  static inline int2 min  (const int2& a, const int2& b)   { return int2{ std::min(a[0], b[0]), std::min(a[1], b[1])}; }
  static inline int2 max  (const int2& a, const int2& b)   { return int2{ std::max(a[0], b[0]), std::max(a[1], b[1])}; }
  static inline int2 clamp(const int2 & u, int a, int b)   { return int2{ clamp(u.x, a, b),     clamp(u.y, a, b)};     }
  
  static inline int2 abs (const int2& a)       { return int2{std::abs(a.M[0]),   std::abs(a.M[1])};   } 
  static inline int2 sign(const int2& a)       { return int2{sign(a.M[0]),       sign(a.M[1])};       }
  
  static inline int extract_0(const int2& a) { return a.M[0]; }
  static inline int extract_1(const int2& a) { return a.M[1]; }

  static inline int2 splat_0(const int2& a)  { return int2{ a.M[0], a.M[0] }; }
  static inline int2 splat_1(const int2& a)  { return int2{ a.M[1], a.M[1] }; }

  static inline int hmin    (const int2& a) { return std::min(a.M[0], a.M[1]); }
  static inline int hmax    (const int2& a) { return std::max(a.M[0], a.M[1]); }
 
  static inline int2 blend(const int2 a, const int2 b, const uint2 mask) { return int2{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y}; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct CVEX_ALIGNED(16) float4
  {
    inline float4() : x(0), y(0), z(0), w(0) {}
    inline float4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    inline explicit float4(float a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
    inline explicit float4(float a) { v = cvex::splat(a); }

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

  static inline float4 operator+(const float a, const float4 b) { return float4(a + b.v); }
  static inline float4 operator-(const float a, const float4 b) { return float4(a - b.v); }
  static inline float4 operator*(const float a, const float4 b) { return float4(a * b.v); }
  static inline float4 operator/(const float a, const float4 b) { return float4(a / b.v); }

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

  static inline float4 cross(const float4& a, const float4& b) { return cvex::cross3(a.v, b.v); } 
  static inline float  dot  (const float4& a, const float4& b) { return cvex::dot4f(a.v, b.v);  }
  static inline float  dot3f(const float4& a, const float4& b) { return cvex::dot3f(a.v, b.v);  }
  static inline float4 dot3v(const float4& a, const float4& b) { return cvex::dot3v(a.v, b.v);  }
  static inline float  dot4f(const float4& a, const float4& b) { return cvex::dot4f(a.v, b.v);  }
  static inline float4 dot4v(const float4& a, const float4& b) { return cvex::dot4v(a.v, b.v);  }
  static inline float  dot4 (const float4& a, const float4& b) { return cvex::dot3f(a.v, b.v);  }
  static inline float  dot3 (const float4& a, const float4& b) { return cvex::dot4f(a.v, b.v);  }
  
  static inline float  length3(const float4& a)  { return cvex::length3f(a.v); }
  static inline float  length4(const float4& a)  { return cvex::length4f(a.v); }
  static inline float  length3f(const float4& a) { return cvex::length3f(a.v); }
  static inline float  length4f(const float4& a) { return cvex::length4f(a.v); }
  static inline float4 length3v(const float4& a) { return cvex::length3v(a.v); }
  static inline float4 length4v(const float4& a) { return cvex::length4v(a.v); }
  static inline float4 normalize (const float4& u) { return u / length4f(u); }
  static inline float4 normalize3(const float4& u) { return u / length3f(u); }

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
  static inline float4 refract(const float4 incidentVec, const float4 normal, float eta)
  {
    float N_dot_I = dot(normal, incidentVec);
    float k = float(1.f) - eta * eta * (float(1.f) - N_dot_I * N_dot_I);
    if (k < float(0.f))
      return float4(0.f);
    else
      return eta * incidentVec - (eta * N_dot_I + std::sqrt(k)) * normal;
  }
   // A floating-point, surface normal vector that is facing the view direction
  static inline float4 faceforward(const float4 N, const float4 I, const float4 Ng) { return dot(I, Ng) < float(0) ? N : float(-1)*N; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct float3
  {
    inline float3() {}
    inline float3(float a, float b, float c) { CVEX_ALIGNED(16) float data[4] = {a,b,c,1.0f};           v = cvex::load(data); }
    inline explicit float3(float a[3])       { CVEX_ALIGNED(16) float data[4] = {a[0],a[1],a[2],11.0f}; v = cvex::load(data); }
    inline explicit float3(float a)          { v = cvex::splat(a); }

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

    inline uint3 operator> (const float3& b) const { return (v > b.v); }
    inline uint3 operator< (const float3& b) const { return (v < b.v); }
    inline uint3 operator>=(const float3& b) const { return (v >= b.v); }
    inline uint3 operator<=(const float3& b) const { return (v <= b.v); }
    inline uint3 operator==(const float3& b) const { return (v == b.v); }
    inline uint3 operator!=(const float3& b) const { return (v != b.v); }

    union
    {
      struct {float x, y, z; };
      float  M[3];
      cvex::vfloat4 v;
    };
  };

  static inline float3 operator+(const float a, const float3 b) { return float3(a + b.v); }
  static inline float3 operator-(const float a, const float3 b) { return float3(a - b.v); }
  static inline float3 operator*(const float a, const float3 b) { return float3(a * b.v); }
  static inline float3 operator/(const float a, const float3 b) { return float3(a / b.v); }

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

  static inline float extract_0(const float3& a_val) { return cvex::extract_0(a_val.v); }
  static inline float extract_1(const float3& a_val) { return cvex::extract_1(a_val.v); }
  static inline float extract_2(const float3& a_val) { return cvex::extract_2(a_val.v); }
  static inline float extract_3(const float3& a_val) { return cvex::extract_3(a_val.v); }

  static inline float3 splat_0(const float3& v)      { return cvex::splat_0(v.v); }
  static inline float3 splat_1(const float3& v)      { return cvex::splat_1(v.v); }
  static inline float3 splat_2(const float3& v)      { return cvex::splat_2(v.v); }  
 
  static inline float hmin(const float3 a_val) { return cvex::hmin3(a_val.v); }
  static inline float hmax(const float3 a_val) { return cvex::hmax3(a_val.v); }

  static inline float3 blend(const float3 a, const float3 b, const uint3 mask) { return cvex::blend(a.v, b.v, mask.v); }
  
  static inline float3 shuffle_xzy(float3 a_src) { return cvex::shuffle_xzyw(a_src.v); }
  static inline float3 shuffle_yxz(float3 a_src) { return cvex::shuffle_yxzw(a_src.v); }
  static inline float3 shuffle_yzx(float3 a_src) { return cvex::shuffle_yzxw(a_src.v); }
  static inline float3 shuffle_zxy(float3 a_src) { return cvex::shuffle_zxyw(a_src.v); }
  static inline float3 shuffle_zyx(float3 a_src) { return cvex::shuffle_zyxw(a_src.v); }
  
  static inline float3 reflect(float3 dir, float3 normal) { return cvex::reflect(dir.v, normal.v); }
  static inline float3 refract(const float3 incidentVec, const float3 normal, float eta)
  {
    float N_dot_I = dot(normal, incidentVec);
    float k = float(1.f) - eta * eta * (float(1.f) - N_dot_I * N_dot_I);
    if (k < float(0.f))
      return float3(0.f);
    else
      return eta * incidentVec - (eta * N_dot_I + std::sqrt(k)) * normal;
  }
  // A floating-point, surface normal vector that is facing the view direction
  static inline float3 faceforward(const float3 N, const float3 I, const float3 Ng) { return dot(I, Ng) < float(0) ? N : float(-1)*N; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  

  struct float2
  {
    inline float2() : x(0), y(0) {}
    inline float2(float a, float b) : x(a), y(b) {}
    //inline float2(const std::initializer_list<float> a_v) { M[0] = a_v.begin()[0]; M[1] = a_v.begin()[1]; }
    inline explicit float2(float a[2]) : x(a[0]), y(a[1]) {}
    inline explicit float2(float a)    : x(a), y(a) {}
   
    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct {float x, y; };
      float M[2];
    };
  };

  static inline float2 operator * (const float2& u, float v) { return float2{u.x * v, u.y * v}; }
  static inline float2 operator / (const float2& u, float v) { return float2{u.x / v, u.y / v}; }
  static inline float2 operator * (float v, const float2& u) { return float2{v * u.x, v * u.y}; }
  static inline float2 operator / (float v, const float2& u) { return float2{v / u.x, v / u.y}; }

  static inline float2 operator + (const float2& u, float v) { return float2{u.x + v, u.y + v}; }
  static inline float2 operator - (const float2& u, float v) { return float2{u.x - v, u.y - v}; }
  static inline float2 operator + (float v, const float2& u) { return float2{v + u.x, v + u.y}; }
  static inline float2 operator - (float v, const float2& u) { return float2{v - u.x, v - u.y}; }

  static inline float2 operator + (const float2& u, const float2& v) { return float2{u.x + v.x, u.y + v.y}; }
  static inline float2 operator - (const float2& u, const float2& v) { return float2{u.x - v.x, u.y - v.y}; }
  static inline float2 operator * (const float2& u, const float2& v) { return float2{u.x * v.x, u.y * v.y}; }
  static inline float2 operator / (const float2& u, const float2& v) { return float2{u.x / v.x, u.y / v.y}; }
  static inline float2 operator - (const float2& v) { return {-v.x, -v.y}; }

  static inline float2& operator += (float2& u, const float2& v) { u.x += v.x; u.y += v.y; return u; }
  static inline float2& operator -= (float2& u, const float2& v) { u.x -= v.x; u.y -= v.y; return u; }
  static inline float2& operator *= (float2& u, const float2& v) { u.x *= v.x; u.y *= v.y; return u; }
  static inline float2& operator /= (float2& u, const float2& v) { u.x /= v.x; u.y /= v.y; return u; }

  static inline float2& operator += (float2& u, float v) { u.x += v; u.y += v; return u; }
  static inline float2& operator -= (float2& u, float v) { u.x -= v; u.y -= v; return u; }
  static inline float2& operator *= (float2& u, float v) { u.x *= v; u.y *= v; return u; }
  static inline float2& operator /= (float2& u, float v) { u.x /= v; u.y /= v; return u; }
  
  static inline uint2 operator> (const float2& a, const float2& b) { return uint2{a[0] > b[0]  ? 0xFFFFFFFF : 0, a[1] > b[1]  ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator< (const float2& a, const float2& b) { return uint2{a[0] < b[0]  ? 0xFFFFFFFF : 0, a[1] < b[1]  ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator>=(const float2& a, const float2& b) { return uint2{a[0] >= b[0] ? 0xFFFFFFFF : 0, a[1] >= b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator<=(const float2& a, const float2& b) { return uint2{a[0] <= b[0] ? 0xFFFFFFFF : 0, a[1] <= b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator==(const float2& a, const float2& b) { return uint2{a[0] == b[0] ? 0xFFFFFFFF : 0, a[1] == b[1] ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator!=(const float2& a, const float2& b) { return uint2{a[0] != b[0] ? 0xFFFFFFFF : 0, a[1] != b[1] ? 0xFFFFFFFF : 0}; }
  
  static inline void store  (float* p, float2 a_val) { memcpy(p, &a_val, sizeof(float)*2); }
  static inline void store_u(float* p, float2 a_val) { memcpy(p, &a_val, sizeof(float)*2); }  

  static inline float2 lerp(const float2& u, const float2& v, float t) { return u + t * (v - u); }
  static inline float2 mix (const float2& u, const float2& v, float t) { return u + t * (v - u); }
  
  static inline float  dot(const float2& u, const float2& v)  { return (u.x*v.x + u.y*v.y); }
  
  static inline float2 min  (const float2& a, const float2& b)   { return float2{ std::min(a[0], b[0]), std::min(a[1], b[1])}; }
  static inline float2 max  (const float2& a, const float2& b)   { return float2{ std::max(a[0], b[0]), std::max(a[1], b[1])}; }
  static inline float2 clamp(const float2 & u, float a, float b) { return float2{ clamp(u.x, a, b),     clamp(u.y, a, b)};     }

  static inline float  length(const float2 & u)    { return sqrtf(SQR(u.x) + SQR(u.y)); }
  static inline float2 normalize(const float2 & u) { return u / length(u); }
  
  static inline float2 floor(float2 a)           { return float2{std::floor(a.M[0]), std::floor(a.M[1])}; }
  static inline float2 ceil(const float2& a)     { return float2{std::ceil(a.M[0]),  std::ceil(a.M[1])};  }
  static inline float2 abs (const float2& a)     { return float2{std::abs(a.M[0]),   std::abs(a.M[1])};   } 
  static inline float2 sign(const float2& a)     { return float2{sign(a.M[0]),       sign(a.M[1])};       }
  static inline float2 rcp (const float2& a)     { return float2{rcp (a.M[0]),       rcp (a.M[1])};       }
  static inline float2 mod (float2 x, float2 y)  { return x - y * floor(x/y);                             }
  static inline float2 fract(float2 x)           { return x - floor(x);                                   }
  static inline float2 sqrt(float2 a)            { return float2{std::sqrt(a.M[0]), std::sqrt(a.M[1])};   }
  static inline float2 inversesqrt(float2 a)     { return 1.0f/sqrt(a);                                   }
  
  static inline float extract_0(const float2& a) { return a.M[0]; }
  static inline float extract_1(const float2& a) { return a.M[1]; }

  static inline float2 splat_0(const float2& a)  { return float2{ a.M[0], a.M[0] }; }
  static inline float2 splat_1(const float2& a)  { return float2{ a.M[1], a.M[1] }; }

  static inline float hmin    (const float2& a) { return std::min(a.M[0], a.M[1]); }
  static inline float hmax    (const float2& a) { return std::max(a.M[0], a.M[1]); }
 
  static inline float2 blend(const float2 a, const float2 b, const uint2 mask) { return float2{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y}; }
  static inline float2 reflect(const float2 dir, const float2 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
  static inline float2 refract(const float2 incidentVec, const float2 normal, float eta)
  {
    float N_dot_I = dot(normal, incidentVec);
    float k = float(1.f) - eta * eta * (float(1.f) - N_dot_I * N_dot_I);
    if (k < float(0.f))
      return float2(0.f);
    else
      return eta * incidentVec - (eta * N_dot_I + std::sqrt(k)) * normal;
  }
  // A floating-point, surface normal vector that is facing the view direction
  static inline float2 faceforward(const float2 N, const float2 I, const float2 Ng) { return dot(I, Ng) < float(0) ? N : float(-1)*N; }

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

    inline explicit float4x4(float A0, float A1, float A2, float A3,
                             float A4, float A5, float A6, float A7,
                             float A8, float A9, float A10, float A11,
                             float A12, float A13, float A14, float A15)
    {
      m_col[0] = float4{ A0, A4, A8,  A12 };
      m_col[1] = float4{ A1, A5, A9,  A13 };
      m_col[2] = float4{ A2, A6, A10, A14 };
      m_col[3] = float4{ A3, A7, A11, A15 };
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

  static inline float  lengthSquare(const float3& u) { return dot(u,u); }
  static inline float  maxcomp     (const float3& u) { return std::max(u.x, std::max(u.y, u.z)); }
  static inline float  mincomp     (const float3& u) { return std::min(u.x, std::min(u.y, u.z)); }

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

  static inline float4 make_float4(float a, float b, float c, float d) { return float4{a,b,c,d}; }
  static inline float3 make_float3(float a, float b, float c)          { return float3{a,b,c};   }
  static inline float2 make_float2(float a, float b)                   { return float2{a,b};   }
  static inline uint4  make_uint4(uint a, uint b, uint c, uint d)      { return uint4{a,b,c,d}; }
  static inline uint3  make_uint3(uint a, uint b, uint c)              { return uint3{a,b,c};   }
  static inline uint2  make_uint2(uint a, uint b)                      { return uint2{a,b};   }
  static inline int4   make_uint4(int a, int b, int c, int d)          { return int4{a,b,c,d}; }
  static inline int3   make_uint3(int a, int b, int c)                 { return int3{a,b,c};   }
  static inline int2   make_uint2(int a, int b)                        { return int2{a,b};     }
  
  static inline float3 to_float3(float4 f4)         { return float3(f4.v); }
  static inline float4 to_float4(float3 v, float w) { return cvex::blend(v.v, cvex::splat(w), cvex::vuint4{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 }); }
  static inline uint3  to_uint3(uint4 f4)           { return uint3(f4.v); }
  static inline uint4  to_uint4(uint3 v, uint w)    { return cvex::blend(v.v, cvex::splat(w), cvex::vuint4{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 }); }
  static inline int3   to_int3(int4 f4)             { return int3(f4.v); }
  static inline int4   to_int4(int3 v, int w)       { return cvex::blend(v.v, cvex::splat(w), cvex::vuint4{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0 }); }

  static inline int4   to_int32  (const float4 a) { return cvex::to_int32(a.v);  }
  static inline int4   to_int32  (const uint4 a)  { return cvex::to_int32(a.v); }
  static inline uint4  to_uint32 (const float4 a) { return cvex::to_uint32(a.v); }
  static inline uint4  to_uint32 (const int4 a)   { return cvex::to_uint32(a.v); }

  static inline float4 to_float32(const  int4& a)  { return cvex::to_float32(a.v); }
  static inline float4 to_float32(const uint4& a)  { return cvex::to_float32(a.v); }

  static inline float4 as_float32(const int4 a_val)   { return cvex::as_float32(a_val.v); }
  static inline float4 as_float32(const uint4 a_val)  { return cvex::as_float32(a_val.v); }
  static inline int4   as_int32  (const float4 a_val) { return cvex::as_int32  (a_val.v); }
  static inline uint4  as_uint32 (const float4 a_val) { return cvex::as_uint32 (a_val.v); }

  static inline int3   to_int32  (const float3 a) { return cvex::to_int32(a.v);  }
  static inline uint3  to_uint32 (const float3 a) { return cvex::to_uint32(a.v); }
  static inline float3 to_float32(const  int3 a)  { return cvex::to_float32(a.v); }
  static inline float3 to_float32(const uint3 a)  { return cvex::to_float32(a.v); }
  
  static inline float3 as_float32(const int3 a_val)   { return cvex::as_float32(a_val.v); }
  static inline float3 as_float32(const uint3 a_val)  { return cvex::as_float32(a_val.v); }
  static inline int3   as_int32  (const float3 a_val) { return cvex::as_int32  (a_val.v); }
  static inline uint3  as_uint32 (const float3 a_val) { return cvex::as_uint32 (a_val.v); }

  static inline int2   to_int32  (const float2 a) { return int2  {int(a.x),   int(a.y)}; }
  static inline uint2  to_uint32 (const float2 a) { return uint2 {uint(a.x),  uint(a.y)}; }
  static inline float2 to_float32(const  int2 a)  { return float2{float(a.x), float(a.y)}; }
  static inline float2 to_float32(const uint2 a)  { return float2{float(a.x), float(a.y)}; }
  
  static inline float2 as_float32(const int2   a_val) { float2 res; memcpy(&res, &a_val, sizeof(float)*2); return res; }
  static inline float2 as_float32(const uint2  a_val) { float2 res; memcpy(&res, &a_val, sizeof(float)*2); return res; }
  static inline int2   as_int32  (const float2 a_val) { int2 res;   memcpy(&res, &a_val, sizeof(int)*2);   return res; }
  static inline uint2  as_uint32 (const float2 a_val) { uint2 res;  memcpy(&res, &a_val, sizeof(uint)*2);  return res; }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  struct ushort4
  {
    ushort4() :x(0), y(0), z(0), w(0) {}
    ushort4(unsigned short a, unsigned short b, unsigned short c, unsigned short d) : x(a), y(b), z(c), w(d) {}

    unsigned short x, y, z, w;
  };  

  struct ushort2
  {
    ushort2() : x(0), y(0) {}
    ushort2(unsigned short a, unsigned short b) : x(a), y(b) {}

    unsigned short x, y;
  };

  struct uchar4
  {
    inline uchar4() : x(0), y(0), z(0), w(0) {}
    inline uchar4(unsigned char a, unsigned char b, unsigned char c, unsigned char d) : x(a), y(b), z(c), w(d) {}
    inline explicit uchar4(unsigned char a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}
    
    inline unsigned char& operator[](uint i)       { return M[i]; }
    inline unsigned char  operator[](uint i) const { return M[i]; }
    
    union
    {
      struct {unsigned char x, y, z, w; };
      unsigned char  M[4];
    };
  };

};

#endif
