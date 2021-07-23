#include <cmath>
#include <limits>           // for std::numeric_limits
#include <cstring>          // for memcpy

#ifdef M_PI
#undef M_PI // same if we have such macro some-where else ...
#endif

#ifdef min 
#undef min // if we on windows, need thid due to macro definitions in windows.h; same if we have such macro some-where else.
#endif

#ifdef max
#undef min // if we on windows, need thid due to macro definitions in windows.h; same if we have such macro some-where else.
#endif

#include <algorithm>        // for std::min/std::max 
#include <initializer_list> //

#if defined(_MSC_VER)
#define CVEX_ALIGNED(x) __declspec(align(x))
#else
#if defined(__GNUC__)
#define CVEX_ALIGNED(x) __attribute__ ((aligned(x)))
#endif
#endif

namespace LiteMath
{ 
  typedef unsigned int   uint;
  typedef unsigned short ushort;
  typedef unsigned char  uchar;

  constexpr float EPSILON      = 1e-6f;
  constexpr float INF_POSITIVE = +std::numeric_limits<float>::infinity();
  constexpr float INF_NEGATIVE = -std::numeric_limits<float>::infinity();
  
  constexpr float DEG_TO_RAD   = float(3.14159265358979323846f) / 180.0f;
  constexpr float M_PI         = float(3.14159265358979323846f);
  constexpr float M_TWOPI      = M_PI*2.0f;
  constexpr float INV_PI       = 1.0f/M_PI;
  constexpr float INV_TWOPI    = 1.0f/(2.0f*M_PI);

  using std::min;
  using std::max;
  using std::sqrt;
  using std::abs;

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

  static inline int   as_int32(float x)  { return as_int(x);    }
  static inline uint  as_uint32(float x) { return as_uint32(x); }
  static inline float as_float32(int x)  { return as_float(x);  }

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

  struct uint4
  {
    inline uint4() : x(0), y(0), z(0), w(0) {}
    inline uint4(uint a_x, uint a_y, uint a_z, uint a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit uint4(uint a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit uint4(const uint a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline uint& operator[](int i)       { return M[i]; }
    inline uint  operator[](int i) const { return M[i]; }

    union
    {
      struct { uint x, y, z, w; };
      uint M[4];
    };
  };

  static inline uint4 operator+(const uint4 a, const uint4 b) { return uint4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
  static inline uint4 operator-(const uint4 a, const uint4 b) { return uint4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
  static inline uint4 operator*(const uint4 a, const uint4 b) { return uint4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
  static inline uint4 operator/(const uint4 a, const uint4 b) { return uint4{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w}; }

  static inline uint4 operator * (const uint4 a, uint b) { return uint4{a.x * b, a.y * b, a.z * b, a.w * b}; }
  static inline uint4 operator / (const uint4 a, uint b) { return uint4{a.x / b, a.y / b, a.z / b, a.w / b}; }
  static inline uint4 operator * (uint a, const uint4 b) { return uint4{a * b.x, a * b.y, a * b.z, a * b.w}; }
  static inline uint4 operator / (uint a, const uint4 b) { return uint4{a / b.x, a / b.y, a / b.z, a / b.w}; }

  static inline uint4 operator + (const uint4 a, uint b) { return uint4{a.x + b, a.y + b, a.z + b, a.w + b}; }
  static inline uint4 operator - (const uint4 a, uint b) { return uint4{a.x - b, a.y - b, a.z - b, a.w - b}; }
  static inline uint4 operator + (uint a, const uint4 b) { return uint4{a + b.x, a + b.y, a + b.z, a + b.w}; }
  static inline uint4 operator - (uint a, const uint4 b) { return uint4{a - b.x, a - b.y, a - b.z, a - b.w}; }

  static inline uint4& operator *= (uint4& a, const uint4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;  return a; }
  static inline uint4& operator /= (uint4& a, const uint4 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;  return a; }
  static inline uint4& operator *= (uint4& a, uint b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b;  return a; }
  static inline uint4& operator /= (uint4& a, uint b) { a.x /= b; a.y /= b; a.z /= b; a.w /= b;  return a; }

  static inline uint4& operator += (uint4& a, const uint4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;  return a; }
  static inline uint4& operator -= (uint4& a, const uint4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;  return a; }
  static inline uint4& operator += (uint4& a, uint b) { a.x += b; a.y += b; a.z += b; a.w += b;  return a; }
  static inline uint4& operator -= (uint4& a, uint b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b;  return a; }

  static inline uint4 operator> (const uint4 a, const uint4 b) { return uint4{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0, a.w >  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator< (const uint4 a, const uint4 b) { return uint4{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0, a.w <  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator>=(const uint4 a, const uint4 b) { return uint4{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0, a.w >= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator<=(const uint4 a, const uint4 b) { return uint4{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0, a.w <= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator==(const uint4 a, const uint4 b) { return uint4{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0, a.w == b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator!=(const uint4 a, const uint4 b) { return uint4{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0, a.w != b.w ? 0xFFFFFFFF : 0}; }
 
  static inline uint4 operator& (const uint4 a, const uint4 b) { return uint4{a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w}; }
  static inline uint4 operator| (const uint4 a, const uint4 b) { return uint4{a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w}; }
  static inline uint4 operator~ (const uint4 a)                { return uint4{~a.x, ~a.y, ~a.z, ~a.w}; }
  static inline uint4 operator>>(const uint4 a, uint b) { return uint4{a.x >> b, a.y >> b, a.z >> b, a.w >> b}; }
  static inline uint4 operator<<(const uint4 a, uint b) { return uint4{a.x << b, a.y << b, a.z << b, a.w << b}; }
 
  static inline bool all_of(const uint4 a) { return (a.x != 0 && a.y != 0 && a.z != 0 && a.w != 0); } 
  static inline bool any_of(const uint4 a) { return (a.x != 0 || a.y != 0 || a.z != 0 || a.w != 0); } 
 

  static inline void store  (uint* p, const uint4 a_val) { memcpy(p, &a_val, sizeof(uint)*4); }
  static inline void store_u(uint* p, const uint4 a_val) { memcpy(p, &a_val, sizeof(uint)*4); }  


  static inline uint4 min  (const uint4 a, const uint4 b) { return uint4{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)}; }
  static inline uint4 max  (const uint4 a, const uint4 b) { return uint4{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)}; }
  static inline uint4 clamp(const uint4 u, const uint4 a, const uint4 b) { return uint4{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z), clamp(u.w, a.w, b.w)}; }
  static inline uint4 clamp(const uint4 u, uint a, uint b) { return uint4{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b), clamp(u.w, a, b)}; }
  
  static inline  uint dot(const uint4 a, const uint4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
  static inline  uint length(const uint4 a) { return std::sqrt(dot(a,a)); }
  static inline  uint4 normalize(const uint4 a) { uint lenInv = uint(1)/length(a); return a*lenInv; }

  static inline uint  dot3(const uint4 a, const uint4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline uint  dot4(const uint4 a, const uint4 b)  { return dot(a,b); } 
  static inline uint4 blend(const uint4 a, const uint4 b, const uint4 mask) { return uint4{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y, (mask.z == 0) ? b.z : a.z, (mask.w == 0) ? b.w : a.w}; }

  static inline uint extract_0(const uint4 a) { return a.x; } 
  static inline uint extract_1(const uint4 a) { return a.y; } 
  static inline uint extract_2(const uint4 a) { return a.z; } 
  static inline uint extract_3(const uint4 a) { return a.w; } 

  static inline uint4 splat_0(const uint4 a) { return uint4{a.x, a.x, a.x, a.x}; } 
  static inline uint4 splat_1(const uint4 a) { return uint4{a.y, a.y, a.y, a.y}; } 
  static inline uint4 splat_2(const uint4 a) { return uint4{a.z, a.z, a.z, a.z}; } 
  static inline uint4 splat_3(const uint4 a) { return uint4{a.w, a.w, a.w, a.w}; } 

  static inline uint hmin(const uint4 a)  { return std::min(std::min(a.x, a.y), std::min(a.z, a.w) ); }
  static inline uint hmax(const uint4 a)  { return std::max(std::max(a.x, a.y), std::max(a.z, a.w) ); }
  static inline uint hmin3(const uint4 a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline uint hmax3(const uint4 a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline uint4 shuffle_xzyw(uint4 a) { return uint4{a.x, a.z, a.y, a.w}; }
  static inline uint4 shuffle_yxzw(uint4 a) { return uint4{a.y, a.x, a.z, a.w}; }
  static inline uint4 shuffle_yzxw(uint4 a) { return uint4{a.y, a.z, a.x, a.w}; }
  static inline uint4 shuffle_zxyw(uint4 a) { return uint4{a.z, a.x, a.y, a.w}; }
  static inline uint4 shuffle_zyxw(uint4 a) { return uint4{a.z, a.y, a.x, a.w}; }
  static inline uint4 shuffle_xyxy(uint4 a) { return uint4{a.x, a.y, a.x, a.y}; }
  static inline uint4 shuffle_zwzw(uint4 a) { return uint4{a.z, a.w, a.z, a.w}; }
  static inline uint4 cross3(const uint4 a, const uint4 b) 
  {
    const uint4 a_yzx = shuffle_yzxw(a);
    const uint4 b_yzx = shuffle_yzxw(b);
    return shuffle_yzxw(a*b_yzx - a_yzx*b);
  }
  static inline uint4 cross(const uint4 a, const uint4 b) { return cross3(a,b); }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct int4
  {
    inline int4() : x(0), y(0), z(0), w(0) {}
    inline int4(int a_x, int a_y, int a_z, int a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit int4(int a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit int4(const int a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline int& operator[](int i)       { return M[i]; }
    inline int  operator[](int i) const { return M[i]; }

    union
    {
      struct { int x, y, z, w; };
      int M[4];
    };
  };

  static inline int4 operator+(const int4 a, const int4 b) { return int4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
  static inline int4 operator-(const int4 a, const int4 b) { return int4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
  static inline int4 operator*(const int4 a, const int4 b) { return int4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
  static inline int4 operator/(const int4 a, const int4 b) { return int4{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w}; }

  static inline int4 operator * (const int4 a, int b) { return int4{a.x * b, a.y * b, a.z * b, a.w * b}; }
  static inline int4 operator / (const int4 a, int b) { return int4{a.x / b, a.y / b, a.z / b, a.w / b}; }
  static inline int4 operator * (int a, const int4 b) { return int4{a * b.x, a * b.y, a * b.z, a * b.w}; }
  static inline int4 operator / (int a, const int4 b) { return int4{a / b.x, a / b.y, a / b.z, a / b.w}; }

  static inline int4 operator + (const int4 a, int b) { return int4{a.x + b, a.y + b, a.z + b, a.w + b}; }
  static inline int4 operator - (const int4 a, int b) { return int4{a.x - b, a.y - b, a.z - b, a.w - b}; }
  static inline int4 operator + (int a, const int4 b) { return int4{a + b.x, a + b.y, a + b.z, a + b.w}; }
  static inline int4 operator - (int a, const int4 b) { return int4{a - b.x, a - b.y, a - b.z, a - b.w}; }

  static inline int4& operator *= (int4& a, const int4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;  return a; }
  static inline int4& operator /= (int4& a, const int4 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;  return a; }
  static inline int4& operator *= (int4& a, int b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b;  return a; }
  static inline int4& operator /= (int4& a, int b) { a.x /= b; a.y /= b; a.z /= b; a.w /= b;  return a; }

  static inline int4& operator += (int4& a, const int4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;  return a; }
  static inline int4& operator -= (int4& a, const int4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;  return a; }
  static inline int4& operator += (int4& a, int b) { a.x += b; a.y += b; a.z += b; a.w += b;  return a; }
  static inline int4& operator -= (int4& a, int b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b;  return a; }

  static inline uint4 operator> (const int4 a, const int4 b) { return uint4{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0, a.w >  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator< (const int4 a, const int4 b) { return uint4{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0, a.w <  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator>=(const int4 a, const int4 b) { return uint4{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0, a.w >= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator<=(const int4 a, const int4 b) { return uint4{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0, a.w <= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator==(const int4 a, const int4 b) { return uint4{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0, a.w == b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator!=(const int4 a, const int4 b) { return uint4{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0, a.w != b.w ? 0xFFFFFFFF : 0}; }
 
  static inline int4 operator& (const int4 a, const int4 b) { return int4{a.x & b.x, a.y & b.y, a.z & b.z, a.w & b.w}; }
  static inline int4 operator| (const int4 a, const int4 b) { return int4{a.x | b.x, a.y | b.y, a.z | b.z, a.w | b.w}; }
  static inline int4 operator~ (const int4 a)                { return int4{~a.x, ~a.y, ~a.z, ~a.w}; }
  static inline int4 operator>>(const int4 a, int b) { return int4{a.x >> b, a.y >> b, a.z >> b, a.w >> b}; }
  static inline int4 operator<<(const int4 a, int b) { return int4{a.x << b, a.y << b, a.z << b, a.w << b}; }
 
  static inline bool all_of(const int4 a) { return (a.x != 0 && a.y != 0 && a.z != 0 && a.w != 0); } 
  static inline bool any_of(const int4 a) { return (a.x != 0 || a.y != 0 || a.z != 0 || a.w != 0); } 
 

  static inline void store  (int* p, const int4 a_val) { memcpy(p, &a_val, sizeof(int)*4); }
  static inline void store_u(int* p, const int4 a_val) { memcpy(p, &a_val, sizeof(int)*4); }  


  static inline int4 min  (const int4 a, const int4 b) { return int4{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)}; }
  static inline int4 max  (const int4 a, const int4 b) { return int4{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)}; }
  static inline int4 clamp(const int4 u, const int4 a, const int4 b) { return int4{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z), clamp(u.w, a.w, b.w)}; }
  static inline int4 clamp(const int4 u, int a, int b) { return int4{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b), clamp(u.w, a, b)}; }

  static inline int4 abs (const int4 a) { return int4{std::abs(a.x), std::abs(a.y), std::abs(a.z), std::abs(a.w)}; } 
  static inline int4 sign(const int4 a) { return int4{sign(a.x), sign(a.y), sign(a.z), sign(a.w)}; }
  
  static inline  int dot(const int4 a, const int4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
  static inline  int length(const int4 a) { return std::sqrt(dot(a,a)); }
  static inline  int4 normalize(const int4 a) { int lenInv = int(1)/length(a); return a*lenInv; }

  static inline int  dot3(const int4 a, const int4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline int  dot4(const int4 a, const int4 b)  { return dot(a,b); } 
  static inline int4 blend(const int4 a, const int4 b, const uint4 mask) { return int4{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y, (mask.z == 0) ? b.z : a.z, (mask.w == 0) ? b.w : a.w}; }

  static inline int extract_0(const int4 a) { return a.x; } 
  static inline int extract_1(const int4 a) { return a.y; } 
  static inline int extract_2(const int4 a) { return a.z; } 
  static inline int extract_3(const int4 a) { return a.w; } 

  static inline int4 splat_0(const int4 a) { return int4{a.x, a.x, a.x, a.x}; } 
  static inline int4 splat_1(const int4 a) { return int4{a.y, a.y, a.y, a.y}; } 
  static inline int4 splat_2(const int4 a) { return int4{a.z, a.z, a.z, a.z}; } 
  static inline int4 splat_3(const int4 a) { return int4{a.w, a.w, a.w, a.w}; } 

  static inline int hmin(const int4 a)  { return std::min(std::min(a.x, a.y), std::min(a.z, a.w) ); }
  static inline int hmax(const int4 a)  { return std::max(std::max(a.x, a.y), std::max(a.z, a.w) ); }
  static inline int hmin3(const int4 a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline int hmax3(const int4 a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline int4 shuffle_xzyw(int4 a) { return int4{a.x, a.z, a.y, a.w}; }
  static inline int4 shuffle_yxzw(int4 a) { return int4{a.y, a.x, a.z, a.w}; }
  static inline int4 shuffle_yzxw(int4 a) { return int4{a.y, a.z, a.x, a.w}; }
  static inline int4 shuffle_zxyw(int4 a) { return int4{a.z, a.x, a.y, a.w}; }
  static inline int4 shuffle_zyxw(int4 a) { return int4{a.z, a.y, a.x, a.w}; }
  static inline int4 shuffle_xyxy(int4 a) { return int4{a.x, a.y, a.x, a.y}; }
  static inline int4 shuffle_zwzw(int4 a) { return int4{a.z, a.w, a.z, a.w}; }
  static inline int4 cross3(const int4 a, const int4 b) 
  {
    const int4 a_yzx = shuffle_yzxw(a);
    const int4 b_yzx = shuffle_yzxw(b);
    return shuffle_yzxw(a*b_yzx - a_yzx*b);
  }
  static inline int4 cross(const int4 a, const int4 b) { return cross3(a,b); }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct float4
  {
    inline float4() : x(0), y(0), z(0), w(0) {}
    inline float4(float a_x, float a_y, float a_z, float a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit float4(float a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit float4(const float a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct { float x, y, z, w; };
      float M[4];
    };
  };

  static inline float4 operator+(const float4 a, const float4 b) { return float4{a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w}; }
  static inline float4 operator-(const float4 a, const float4 b) { return float4{a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w}; }
  static inline float4 operator*(const float4 a, const float4 b) { return float4{a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w}; }
  static inline float4 operator/(const float4 a, const float4 b) { return float4{a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w}; }

  static inline float4 operator * (const float4 a, float b) { return float4{a.x * b, a.y * b, a.z * b, a.w * b}; }
  static inline float4 operator / (const float4 a, float b) { return float4{a.x / b, a.y / b, a.z / b, a.w / b}; }
  static inline float4 operator * (float a, const float4 b) { return float4{a * b.x, a * b.y, a * b.z, a * b.w}; }
  static inline float4 operator / (float a, const float4 b) { return float4{a / b.x, a / b.y, a / b.z, a / b.w}; }

  static inline float4 operator + (const float4 a, float b) { return float4{a.x + b, a.y + b, a.z + b, a.w + b}; }
  static inline float4 operator - (const float4 a, float b) { return float4{a.x - b, a.y - b, a.z - b, a.w - b}; }
  static inline float4 operator + (float a, const float4 b) { return float4{a + b.x, a + b.y, a + b.z, a + b.w}; }
  static inline float4 operator - (float a, const float4 b) { return float4{a - b.x, a - b.y, a - b.z, a - b.w}; }

  static inline float4& operator *= (float4& a, const float4 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;  return a; }
  static inline float4& operator /= (float4& a, const float4 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z; a.w /= b.w;  return a; }
  static inline float4& operator *= (float4& a, float b) { a.x *= b; a.y *= b; a.z *= b; a.w *= b;  return a; }
  static inline float4& operator /= (float4& a, float b) { a.x /= b; a.y /= b; a.z /= b; a.w /= b;  return a; }

  static inline float4& operator += (float4& a, const float4 b) { a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;  return a; }
  static inline float4& operator -= (float4& a, const float4 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;  return a; }
  static inline float4& operator += (float4& a, float b) { a.x += b; a.y += b; a.z += b; a.w += b;  return a; }
  static inline float4& operator -= (float4& a, float b) { a.x -= b; a.y -= b; a.z -= b; a.w -= b;  return a; }

  static inline uint4 operator> (const float4 a, const float4 b) { return uint4{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0, a.w >  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator< (const float4 a, const float4 b) { return uint4{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0, a.w <  b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator>=(const float4 a, const float4 b) { return uint4{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0, a.w >= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator<=(const float4 a, const float4 b) { return uint4{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0, a.w <= b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator==(const float4 a, const float4 b) { return uint4{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0, a.w == b.w ? 0xFFFFFFFF : 0}; }
  static inline uint4 operator!=(const float4 a, const float4 b) { return uint4{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0, a.w != b.w ? 0xFFFFFFFF : 0}; }

  static inline void store  (float* p, const float4 a_val) { memcpy(p, &a_val, sizeof(float)*4); }
  static inline void store_u(float* p, const float4 a_val) { memcpy(p, &a_val, sizeof(float)*4); }  


  static inline float4 min  (const float4 a, const float4 b) { return float4{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z), std::min(a.w, b.w)}; }
  static inline float4 max  (const float4 a, const float4 b) { return float4{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z), std::max(a.w, b.w)}; }
  static inline float4 clamp(const float4 u, const float4 a, const float4 b) { return float4{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z), clamp(u.w, a.w, b.w)}; }
  static inline float4 clamp(const float4 u, float a, float b) { return float4{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b), clamp(u.w, a, b)}; }

  static inline float4 abs (const float4 a) { return float4{std::abs(a.x), std::abs(a.y), std::abs(a.z), std::abs(a.w)}; } 
  static inline float4 sign(const float4 a) { return float4{sign(a.x), sign(a.y), sign(a.z), sign(a.w)}; }

  static inline float4 lerp(const float4 a, const float4 b, float t) { return a + t * (b - a); }
  static inline float4 mix (const float4 a, const float4 b, float t) { return a + t * (b - a); }
  static inline float4 floor(const float4 a)                { return float4{std::floor(a.x), std::floor(a.y), std::floor(a.z), std::floor(a.w)}; }
  static inline float4 ceil(const float4 a)                 { return float4{std::ceil(a.x), std::ceil(a.y), std::ceil(a.z), std::ceil(a.w)}; }
  static inline float4 rcp (const float4 a)                 { return float4{1.0f/a.x, 1.0f/a.y, 1.0f/a.z, 1.0f/a.w}; }
  static inline float4 mod (const float4 x, const float4 y) { return x - y * floor(x/y); }
  static inline float4 fract(const float4 x)                { return x - floor(x); }
  static inline float4 sqrt(const float4 a)                 { return float4{std::sqrt(a.x), std::sqrt(a.y), std::sqrt(a.z), std::sqrt(a.w)}; }
  static inline float4 inversesqrt(const float4 a)          { return 1.0f/sqrt(a); }
  
  static inline  float dot(const float4 a, const float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w; }
  static inline  float length(const float4 a) { return std::sqrt(dot(a,a)); }
  static inline  float4 normalize(const float4 a) { float lenInv = float(1)/length(a); return a*lenInv; }

  static inline float  dot3(const float4 a, const float4 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline float  dot4(const float4 a, const float4 b)  { return dot(a,b); } 
  static inline float  dot3f(const float4 a, const float4 b) { return dot3(a,b); }
  static inline float  dot4f(const float4 a, const float4 b) { return dot(a,b); }
  static inline float4 dot3v(const float4 a, const float4 b) { float res = dot3(a,b); return float4(res); }
  static inline float4 dot4v(const float4 a, const float4 b) { float res = dot(a,b);  return float4(res); }

  static inline float length3(const float4 a)  { return std::sqrt(dot3(a,a)); }
  static inline float length3f(const float4 a) { return std::sqrt(dot3(a,a)); }
  static inline float length4(const float4 a)  { return std::sqrt(dot4(a,a)); }
  static inline float length4f(const float4 a) { return std::sqrt(dot4(a,a)); }
  static inline float4 length3v(const float4 a) { float res = std::sqrt(dot3(a,a)); return float4(res); }
  static inline float4 length4v(const float4 a) { float res = std::sqrt(dot4(a,a)); return float4(res); }
  static inline float4 normalize3(const float4 a) { float lenInv = float(1)/length3(a); return a*lenInv; }

  static inline float4 blend(const float4 a, const float4 b, const uint4 mask) { return float4{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y, (mask.z == 0) ? b.z : a.z, (mask.w == 0) ? b.w : a.w}; }

  static inline float extract_0(const float4 a) { return a.x; } 
  static inline float extract_1(const float4 a) { return a.y; } 
  static inline float extract_2(const float4 a) { return a.z; } 
  static inline float extract_3(const float4 a) { return a.w; } 

  static inline float4 splat_0(const float4 a) { return float4{a.x, a.x, a.x, a.x}; } 
  static inline float4 splat_1(const float4 a) { return float4{a.y, a.y, a.y, a.y}; } 
  static inline float4 splat_2(const float4 a) { return float4{a.z, a.z, a.z, a.z}; } 
  static inline float4 splat_3(const float4 a) { return float4{a.w, a.w, a.w, a.w}; } 

  static inline float hmin(const float4 a)  { return std::min(std::min(a.x, a.y), std::min(a.z, a.w) ); }
  static inline float hmax(const float4 a)  { return std::max(std::max(a.x, a.y), std::max(a.z, a.w) ); }
  static inline float hmin3(const float4 a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline float hmax3(const float4 a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline float4 shuffle_xzyw(float4 a) { return float4{a.x, a.z, a.y, a.w}; }
  static inline float4 shuffle_yxzw(float4 a) { return float4{a.y, a.x, a.z, a.w}; }
  static inline float4 shuffle_yzxw(float4 a) { return float4{a.y, a.z, a.x, a.w}; }
  static inline float4 shuffle_zxyw(float4 a) { return float4{a.z, a.x, a.y, a.w}; }
  static inline float4 shuffle_zyxw(float4 a) { return float4{a.z, a.y, a.x, a.w}; }
  static inline float4 shuffle_xyxy(float4 a) { return float4{a.x, a.y, a.x, a.y}; }
  static inline float4 shuffle_zwzw(float4 a) { return float4{a.z, a.w, a.z, a.w}; }
  static inline float4 cross3(const float4 a, const float4 b) 
  {
    const float4 a_yzx = shuffle_yzxw(a);
    const float4 b_yzx = shuffle_yzxw(b);
    return shuffle_yzxw(a*b_yzx - a_yzx*b);
  }
  static inline float4 cross(const float4 a, const float4 b) { return cross3(a,b); }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct uint3
  {
    inline uint3() : x(0), y(0), z(0) {}
    inline uint3(uint a_x, uint a_y, uint a_z) : x(a_x), y(a_y), z(a_z) {}
    inline explicit uint3(uint a_val) : x(a_val), y(a_val), z(a_val) {}
    inline explicit uint3(const uint a[3]) : x(a[0]), y(a[1]), z(a[2]) {}

    inline uint& operator[](int i)       { return M[i]; }
    inline uint  operator[](int i) const { return M[i]; }

    union
    {
      struct { uint x, y, z; };
      uint M[3];
    };
  };

  static inline uint3 operator+(const uint3 a, const uint3 b) { return uint3{a.x + b.x, a.y + b.y, a.z + b.z}; }
  static inline uint3 operator-(const uint3 a, const uint3 b) { return uint3{a.x - b.x, a.y - b.y, a.z - b.z}; }
  static inline uint3 operator*(const uint3 a, const uint3 b) { return uint3{a.x * b.x, a.y * b.y, a.z * b.z}; }
  static inline uint3 operator/(const uint3 a, const uint3 b) { return uint3{a.x / b.x, a.y / b.y, a.z / b.z}; }

  static inline uint3 operator * (const uint3 a, uint b) { return uint3{a.x * b, a.y * b, a.z * b}; }
  static inline uint3 operator / (const uint3 a, uint b) { return uint3{a.x / b, a.y / b, a.z / b}; }
  static inline uint3 operator * (uint a, const uint3 b) { return uint3{a * b.x, a * b.y, a * b.z}; }
  static inline uint3 operator / (uint a, const uint3 b) { return uint3{a / b.x, a / b.y, a / b.z}; }

  static inline uint3 operator + (const uint3 a, uint b) { return uint3{a.x + b, a.y + b, a.z + b}; }
  static inline uint3 operator - (const uint3 a, uint b) { return uint3{a.x - b, a.y - b, a.z - b}; }
  static inline uint3 operator + (uint a, const uint3 b) { return uint3{a + b.x, a + b.y, a + b.z}; }
  static inline uint3 operator - (uint a, const uint3 b) { return uint3{a - b.x, a - b.y, a - b.z}; }

  static inline uint3& operator *= (uint3& a, const uint3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z;  return a; }
  static inline uint3& operator /= (uint3& a, const uint3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z;  return a; }
  static inline uint3& operator *= (uint3& a, uint b) { a.x *= b; a.y *= b; a.z *= b;  return a; }
  static inline uint3& operator /= (uint3& a, uint b) { a.x /= b; a.y /= b; a.z /= b;  return a; }

  static inline uint3& operator += (uint3& a, const uint3 b) { a.x += b.x; a.y += b.y; a.z += b.z;  return a; }
  static inline uint3& operator -= (uint3& a, const uint3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z;  return a; }
  static inline uint3& operator += (uint3& a, uint b) { a.x += b; a.y += b; a.z += b;  return a; }
  static inline uint3& operator -= (uint3& a, uint b) { a.x -= b; a.y -= b; a.z -= b;  return a; }

  static inline uint3 operator> (const uint3 a, const uint3 b) { return uint3{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator< (const uint3 a, const uint3 b) { return uint3{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator>=(const uint3 a, const uint3 b) { return uint3{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator<=(const uint3 a, const uint3 b) { return uint3{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator==(const uint3 a, const uint3 b) { return uint3{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator!=(const uint3 a, const uint3 b) { return uint3{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0}; }
 
  static inline uint3 operator& (const uint3 a, const uint3 b) { return uint3{a.x & b.x, a.y & b.y, a.z & b.z}; }
  static inline uint3 operator| (const uint3 a, const uint3 b) { return uint3{a.x | b.x, a.y | b.y, a.z | b.z}; }
  static inline uint3 operator~ (const uint3 a)                { return uint3{~a.x, ~a.y, ~a.z}; }
  static inline uint3 operator>>(const uint3 a, uint b) { return uint3{a.x >> b, a.y >> b, a.z >> b}; }
  static inline uint3 operator<<(const uint3 a, uint b) { return uint3{a.x << b, a.y << b, a.z << b}; }
 
  static inline bool all_of(const uint3 a) { return (a.x != 0 && a.y != 0 && a.z != 0); } 
  static inline bool any_of(const uint3 a) { return (a.x != 0 || a.y != 0 || a.z != 0); } 
 

  static inline void store  (uint* p, const uint3 a_val) { memcpy(p, &a_val, sizeof(uint)*3); }
  static inline void store_u(uint* p, const uint3 a_val) { memcpy(p, &a_val, sizeof(uint)*3); }  


  static inline uint3 min  (const uint3 a, const uint3 b) { return uint3{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
  static inline uint3 max  (const uint3 a, const uint3 b) { return uint3{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
  static inline uint3 clamp(const uint3 u, const uint3 a, const uint3 b) { return uint3{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z)}; }
  static inline uint3 clamp(const uint3 u, uint a, uint b) { return uint3{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)}; }
  
  static inline  uint dot(const uint3 a, const uint3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline  uint length(const uint3 a) { return std::sqrt(dot(a,a)); }
  static inline  uint3 normalize(const uint3 a) { uint lenInv = uint(1)/length(a); return a*lenInv; }

  static inline uint3 blend(const uint3 a, const uint3 b, const uint3 mask) { return uint3{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y, (mask.z == 0) ? b.z : a.z}; }

  static inline uint extract_0(const uint3 a) { return a.x; } 
  static inline uint extract_1(const uint3 a) { return a.y; } 
  static inline uint extract_2(const uint3 a) { return a.z; } 

  static inline uint3 splat_0(const uint3 a) { return uint3{a.x, a.x, a.x}; } 
  static inline uint3 splat_1(const uint3 a) { return uint3{a.y, a.y, a.y}; } 
  static inline uint3 splat_2(const uint3 a) { return uint3{a.z, a.z, a.z}; } 

  static inline uint hmin(const uint3 a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline uint hmax(const uint3 a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline uint3 shuffle_xzy(uint3 a) { return uint3{a.x, a.z, a.y}; }
  static inline uint3 shuffle_yxz(uint3 a) { return uint3{a.y, a.x, a.z}; }
  static inline uint3 shuffle_yzx(uint3 a) { return uint3{a.y, a.z, a.x}; }
  static inline uint3 shuffle_zxy(uint3 a) { return uint3{a.z, a.x, a.y}; }
  static inline uint3 shuffle_zyx(uint3 a) { return uint3{a.z, a.y, a.x}; }
  static inline uint3 cross(const uint3 a, const uint3 b) 
  {
    const uint3 a_yzx = shuffle_yzx(a);
    const uint3 b_yzx = shuffle_yzx(b);
    return shuffle_yzx(a*b_yzx - a_yzx*b);
  }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct int3
  {
    inline int3() : x(0), y(0), z(0) {}
    inline int3(int a_x, int a_y, int a_z) : x(a_x), y(a_y), z(a_z) {}
    inline explicit int3(int a_val) : x(a_val), y(a_val), z(a_val) {}
    inline explicit int3(const int a[3]) : x(a[0]), y(a[1]), z(a[2]) {}

    inline int& operator[](int i)       { return M[i]; }
    inline int  operator[](int i) const { return M[i]; }

    union
    {
      struct { int x, y, z; };
      int M[3];
    };
  };

  static inline int3 operator+(const int3 a, const int3 b) { return int3{a.x + b.x, a.y + b.y, a.z + b.z}; }
  static inline int3 operator-(const int3 a, const int3 b) { return int3{a.x - b.x, a.y - b.y, a.z - b.z}; }
  static inline int3 operator*(const int3 a, const int3 b) { return int3{a.x * b.x, a.y * b.y, a.z * b.z}; }
  static inline int3 operator/(const int3 a, const int3 b) { return int3{a.x / b.x, a.y / b.y, a.z / b.z}; }

  static inline int3 operator * (const int3 a, int b) { return int3{a.x * b, a.y * b, a.z * b}; }
  static inline int3 operator / (const int3 a, int b) { return int3{a.x / b, a.y / b, a.z / b}; }
  static inline int3 operator * (int a, const int3 b) { return int3{a * b.x, a * b.y, a * b.z}; }
  static inline int3 operator / (int a, const int3 b) { return int3{a / b.x, a / b.y, a / b.z}; }

  static inline int3 operator + (const int3 a, int b) { return int3{a.x + b, a.y + b, a.z + b}; }
  static inline int3 operator - (const int3 a, int b) { return int3{a.x - b, a.y - b, a.z - b}; }
  static inline int3 operator + (int a, const int3 b) { return int3{a + b.x, a + b.y, a + b.z}; }
  static inline int3 operator - (int a, const int3 b) { return int3{a - b.x, a - b.y, a - b.z}; }

  static inline int3& operator *= (int3& a, const int3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z;  return a; }
  static inline int3& operator /= (int3& a, const int3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z;  return a; }
  static inline int3& operator *= (int3& a, int b) { a.x *= b; a.y *= b; a.z *= b;  return a; }
  static inline int3& operator /= (int3& a, int b) { a.x /= b; a.y /= b; a.z /= b;  return a; }

  static inline int3& operator += (int3& a, const int3 b) { a.x += b.x; a.y += b.y; a.z += b.z;  return a; }
  static inline int3& operator -= (int3& a, const int3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z;  return a; }
  static inline int3& operator += (int3& a, int b) { a.x += b; a.y += b; a.z += b;  return a; }
  static inline int3& operator -= (int3& a, int b) { a.x -= b; a.y -= b; a.z -= b;  return a; }

  static inline uint3 operator> (const int3 a, const int3 b) { return uint3{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator< (const int3 a, const int3 b) { return uint3{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator>=(const int3 a, const int3 b) { return uint3{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator<=(const int3 a, const int3 b) { return uint3{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator==(const int3 a, const int3 b) { return uint3{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator!=(const int3 a, const int3 b) { return uint3{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0}; }
 
  static inline int3 operator& (const int3 a, const int3 b) { return int3{a.x & b.x, a.y & b.y, a.z & b.z}; }
  static inline int3 operator| (const int3 a, const int3 b) { return int3{a.x | b.x, a.y | b.y, a.z | b.z}; }
  static inline int3 operator~ (const int3 a)                { return int3{~a.x, ~a.y, ~a.z}; }
  static inline int3 operator>>(const int3 a, int b) { return int3{a.x >> b, a.y >> b, a.z >> b}; }
  static inline int3 operator<<(const int3 a, int b) { return int3{a.x << b, a.y << b, a.z << b}; }
 
  static inline bool all_of(const int3 a) { return (a.x != 0 && a.y != 0 && a.z != 0); } 
  static inline bool any_of(const int3 a) { return (a.x != 0 || a.y != 0 || a.z != 0); } 
 

  static inline void store  (int* p, const int3 a_val) { memcpy(p, &a_val, sizeof(int)*3); }
  static inline void store_u(int* p, const int3 a_val) { memcpy(p, &a_val, sizeof(int)*3); }  


  static inline int3 min  (const int3 a, const int3 b) { return int3{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
  static inline int3 max  (const int3 a, const int3 b) { return int3{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
  static inline int3 clamp(const int3 u, const int3 a, const int3 b) { return int3{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z)}; }
  static inline int3 clamp(const int3 u, int a, int b) { return int3{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)}; }

  static inline int3 abs (const int3 a) { return int3{std::abs(a.x), std::abs(a.y), std::abs(a.z)}; } 
  static inline int3 sign(const int3 a) { return int3{sign(a.x), sign(a.y), sign(a.z)}; }
  
  static inline  int dot(const int3 a, const int3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline  int length(const int3 a) { return std::sqrt(dot(a,a)); }
  static inline  int3 normalize(const int3 a) { int lenInv = int(1)/length(a); return a*lenInv; }

  static inline int3 blend(const int3 a, const int3 b, const uint3 mask) { return int3{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y, (mask.z == 0) ? b.z : a.z}; }

  static inline int extract_0(const int3 a) { return a.x; } 
  static inline int extract_1(const int3 a) { return a.y; } 
  static inline int extract_2(const int3 a) { return a.z; } 

  static inline int3 splat_0(const int3 a) { return int3{a.x, a.x, a.x}; } 
  static inline int3 splat_1(const int3 a) { return int3{a.y, a.y, a.y}; } 
  static inline int3 splat_2(const int3 a) { return int3{a.z, a.z, a.z}; } 

  static inline int hmin(const int3 a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline int hmax(const int3 a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline int3 shuffle_xzy(int3 a) { return int3{a.x, a.z, a.y}; }
  static inline int3 shuffle_yxz(int3 a) { return int3{a.y, a.x, a.z}; }
  static inline int3 shuffle_yzx(int3 a) { return int3{a.y, a.z, a.x}; }
  static inline int3 shuffle_zxy(int3 a) { return int3{a.z, a.x, a.y}; }
  static inline int3 shuffle_zyx(int3 a) { return int3{a.z, a.y, a.x}; }
  static inline int3 cross(const int3 a, const int3 b) 
  {
    const int3 a_yzx = shuffle_yzx(a);
    const int3 b_yzx = shuffle_yzx(b);
    return shuffle_yzx(a*b_yzx - a_yzx*b);
  }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct float3
  {
    inline float3() : x(0), y(0), z(0) {}
    inline float3(float a_x, float a_y, float a_z) : x(a_x), y(a_y), z(a_z) {}
    inline explicit float3(float a_val) : x(a_val), y(a_val), z(a_val) {}
    inline explicit float3(const float a[3]) : x(a[0]), y(a[1]), z(a[2]) {}

    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct { float x, y, z; };
      float M[3];
    };
  };

  static inline float3 operator+(const float3 a, const float3 b) { return float3{a.x + b.x, a.y + b.y, a.z + b.z}; }
  static inline float3 operator-(const float3 a, const float3 b) { return float3{a.x - b.x, a.y - b.y, a.z - b.z}; }
  static inline float3 operator*(const float3 a, const float3 b) { return float3{a.x * b.x, a.y * b.y, a.z * b.z}; }
  static inline float3 operator/(const float3 a, const float3 b) { return float3{a.x / b.x, a.y / b.y, a.z / b.z}; }

  static inline float3 operator * (const float3 a, float b) { return float3{a.x * b, a.y * b, a.z * b}; }
  static inline float3 operator / (const float3 a, float b) { return float3{a.x / b, a.y / b, a.z / b}; }
  static inline float3 operator * (float a, const float3 b) { return float3{a * b.x, a * b.y, a * b.z}; }
  static inline float3 operator / (float a, const float3 b) { return float3{a / b.x, a / b.y, a / b.z}; }

  static inline float3 operator + (const float3 a, float b) { return float3{a.x + b, a.y + b, a.z + b}; }
  static inline float3 operator - (const float3 a, float b) { return float3{a.x - b, a.y - b, a.z - b}; }
  static inline float3 operator + (float a, const float3 b) { return float3{a + b.x, a + b.y, a + b.z}; }
  static inline float3 operator - (float a, const float3 b) { return float3{a - b.x, a - b.y, a - b.z}; }

  static inline float3& operator *= (float3& a, const float3 b) { a.x *= b.x; a.y *= b.y; a.z *= b.z;  return a; }
  static inline float3& operator /= (float3& a, const float3 b) { a.x /= b.x; a.y /= b.y; a.z /= b.z;  return a; }
  static inline float3& operator *= (float3& a, float b) { a.x *= b; a.y *= b; a.z *= b;  return a; }
  static inline float3& operator /= (float3& a, float b) { a.x /= b; a.y /= b; a.z /= b;  return a; }

  static inline float3& operator += (float3& a, const float3 b) { a.x += b.x; a.y += b.y; a.z += b.z;  return a; }
  static inline float3& operator -= (float3& a, const float3 b) { a.x -= b.x; a.y -= b.y; a.z -= b.z;  return a; }
  static inline float3& operator += (float3& a, float b) { a.x += b; a.y += b; a.z += b;  return a; }
  static inline float3& operator -= (float3& a, float b) { a.x -= b; a.y -= b; a.z -= b;  return a; }

  static inline uint3 operator> (const float3 a, const float3 b) { return uint3{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0, a.z >  b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator< (const float3 a, const float3 b) { return uint3{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0, a.z <  b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator>=(const float3 a, const float3 b) { return uint3{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0, a.z >= b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator<=(const float3 a, const float3 b) { return uint3{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0, a.z <= b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator==(const float3 a, const float3 b) { return uint3{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0, a.z == b.z ? 0xFFFFFFFF : 0}; }
  static inline uint3 operator!=(const float3 a, const float3 b) { return uint3{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0, a.z != b.z ? 0xFFFFFFFF : 0}; }

  static inline void store  (float* p, const float3 a_val) { memcpy(p, &a_val, sizeof(float)*3); }
  static inline void store_u(float* p, const float3 a_val) { memcpy(p, &a_val, sizeof(float)*3); }  


  static inline float3 min  (const float3 a, const float3 b) { return float3{std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)}; }
  static inline float3 max  (const float3 a, const float3 b) { return float3{std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)}; }
  static inline float3 clamp(const float3 u, const float3 a, const float3 b) { return float3{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y), clamp(u.z, a.z, b.z)}; }
  static inline float3 clamp(const float3 u, float a, float b) { return float3{clamp(u.x, a, b), clamp(u.y, a, b), clamp(u.z, a, b)}; }

  static inline float3 abs (const float3 a) { return float3{std::abs(a.x), std::abs(a.y), std::abs(a.z)}; } 
  static inline float3 sign(const float3 a) { return float3{sign(a.x), sign(a.y), sign(a.z)}; }

  static inline float3 lerp(const float3 a, const float3 b, float t) { return a + t * (b - a); }
  static inline float3 mix (const float3 a, const float3 b, float t) { return a + t * (b - a); }
  static inline float3 floor(const float3 a)                { return float3{std::floor(a.x), std::floor(a.y), std::floor(a.z)}; }
  static inline float3 ceil(const float3 a)                 { return float3{std::ceil(a.x), std::ceil(a.y), std::ceil(a.z)}; }
  static inline float3 rcp (const float3 a)                 { return float3{1.0f/a.x, 1.0f/a.y, 1.0f/a.z}; }
  static inline float3 mod (const float3 x, const float3 y) { return x - y * floor(x/y); }
  static inline float3 fract(const float3 x)                { return x - floor(x); }
  static inline float3 sqrt(const float3 a)                 { return float3{std::sqrt(a.x), std::sqrt(a.y), std::sqrt(a.z)}; }
  static inline float3 inversesqrt(const float3 a)          { return 1.0f/sqrt(a); }
  
  static inline  float dot(const float3 a, const float3 b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline  float length(const float3 a) { return std::sqrt(dot(a,a)); }
  static inline  float3 normalize(const float3 a) { float lenInv = float(1)/length(a); return a*lenInv; }

  static inline float3 blend(const float3 a, const float3 b, const uint3 mask) { return float3{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y, (mask.z == 0) ? b.z : a.z}; }

  static inline float extract_0(const float3 a) { return a.x; } 
  static inline float extract_1(const float3 a) { return a.y; } 
  static inline float extract_2(const float3 a) { return a.z; } 

  static inline float3 splat_0(const float3 a) { return float3{a.x, a.x, a.x}; } 
  static inline float3 splat_1(const float3 a) { return float3{a.y, a.y, a.y}; } 
  static inline float3 splat_2(const float3 a) { return float3{a.z, a.z, a.z}; } 

  static inline float hmin(const float3 a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline float hmax(const float3 a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline float3 shuffle_xzy(float3 a) { return float3{a.x, a.z, a.y}; }
  static inline float3 shuffle_yxz(float3 a) { return float3{a.y, a.x, a.z}; }
  static inline float3 shuffle_yzx(float3 a) { return float3{a.y, a.z, a.x}; }
  static inline float3 shuffle_zxy(float3 a) { return float3{a.z, a.x, a.y}; }
  static inline float3 shuffle_zyx(float3 a) { return float3{a.z, a.y, a.x}; }
  static inline float3 cross(const float3 a, const float3 b) 
  {
    const float3 a_yzx = shuffle_yzx(a);
    const float3 b_yzx = shuffle_yzx(b);
    return shuffle_yzx(a*b_yzx - a_yzx*b);
  }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct uint2
  {
    inline uint2() : x(0), y(0) {}
    inline uint2(uint a_x, uint a_y) : x(a_x), y(a_y) {}
    inline explicit uint2(uint a_val) : x(a_val), y(a_val) {}
    inline explicit uint2(const uint a[2]) : x(a[0]), y(a[1]) {}

    inline uint& operator[](int i)       { return M[i]; }
    inline uint  operator[](int i) const { return M[i]; }

    union
    {
      struct { uint x, y; };
      uint M[2];
    };
  };

  static inline uint2 operator+(const uint2 a, const uint2 b) { return uint2{a.x + b.x, a.y + b.y}; }
  static inline uint2 operator-(const uint2 a, const uint2 b) { return uint2{a.x - b.x, a.y - b.y}; }
  static inline uint2 operator*(const uint2 a, const uint2 b) { return uint2{a.x * b.x, a.y * b.y}; }
  static inline uint2 operator/(const uint2 a, const uint2 b) { return uint2{a.x / b.x, a.y / b.y}; }

  static inline uint2 operator * (const uint2 a, uint b) { return uint2{a.x * b, a.y * b}; }
  static inline uint2 operator / (const uint2 a, uint b) { return uint2{a.x / b, a.y / b}; }
  static inline uint2 operator * (uint a, const uint2 b) { return uint2{a * b.x, a * b.y}; }
  static inline uint2 operator / (uint a, const uint2 b) { return uint2{a / b.x, a / b.y}; }

  static inline uint2 operator + (const uint2 a, uint b) { return uint2{a.x + b, a.y + b}; }
  static inline uint2 operator - (const uint2 a, uint b) { return uint2{a.x - b, a.y - b}; }
  static inline uint2 operator + (uint a, const uint2 b) { return uint2{a + b.x, a + b.y}; }
  static inline uint2 operator - (uint a, const uint2 b) { return uint2{a - b.x, a - b.y}; }

  static inline uint2& operator *= (uint2& a, const uint2 b) { a.x *= b.x; a.y *= b.y;  return a; }
  static inline uint2& operator /= (uint2& a, const uint2 b) { a.x /= b.x; a.y /= b.y;  return a; }
  static inline uint2& operator *= (uint2& a, uint b) { a.x *= b; a.y *= b;  return a; }
  static inline uint2& operator /= (uint2& a, uint b) { a.x /= b; a.y /= b;  return a; }

  static inline uint2& operator += (uint2& a, const uint2 b) { a.x += b.x; a.y += b.y;  return a; }
  static inline uint2& operator -= (uint2& a, const uint2 b) { a.x -= b.x; a.y -= b.y;  return a; }
  static inline uint2& operator += (uint2& a, uint b) { a.x += b; a.y += b;  return a; }
  static inline uint2& operator -= (uint2& a, uint b) { a.x -= b; a.y -= b;  return a; }

  static inline uint2 operator> (const uint2 a, const uint2 b) { return uint2{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator< (const uint2 a, const uint2 b) { return uint2{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator>=(const uint2 a, const uint2 b) { return uint2{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator<=(const uint2 a, const uint2 b) { return uint2{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator==(const uint2 a, const uint2 b) { return uint2{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator!=(const uint2 a, const uint2 b) { return uint2{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0}; }
 
  static inline uint2 operator& (const uint2 a, const uint2 b) { return uint2{a.x & b.x, a.y & b.y}; }
  static inline uint2 operator| (const uint2 a, const uint2 b) { return uint2{a.x | b.x, a.y | b.y}; }
  static inline uint2 operator~ (const uint2 a)                { return uint2{~a.x, ~a.y}; }
  static inline uint2 operator>>(const uint2 a, uint b) { return uint2{a.x >> b, a.y >> b}; }
  static inline uint2 operator<<(const uint2 a, uint b) { return uint2{a.x << b, a.y << b}; }
 
  static inline bool all_of(const uint2 a) { return (a.x != 0 && a.y != 0); } 
  static inline bool any_of(const uint2 a) { return (a.x != 0 || a.y != 0); } 
 

  static inline void store  (uint* p, const uint2 a_val) { memcpy(p, &a_val, sizeof(uint)*2); }
  static inline void store_u(uint* p, const uint2 a_val) { memcpy(p, &a_val, sizeof(uint)*2); }  


  static inline uint2 min  (const uint2 a, const uint2 b) { return uint2{std::min(a.x, b.x), std::min(a.y, b.y)}; }
  static inline uint2 max  (const uint2 a, const uint2 b) { return uint2{std::max(a.x, b.x), std::max(a.y, b.y)}; }
  static inline uint2 clamp(const uint2 u, const uint2 a, const uint2 b) { return uint2{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y)}; }
  static inline uint2 clamp(const uint2 u, uint a, uint b) { return uint2{clamp(u.x, a, b), clamp(u.y, a, b)}; }
  
  static inline  uint dot(const uint2 a, const uint2 b)  { return a.x*b.x + a.y*b.y; }
  static inline  uint length(const uint2 a) { return std::sqrt(dot(a,a)); }
  static inline  uint2 normalize(const uint2 a) { uint lenInv = uint(1)/length(a); return a*lenInv; }

  static inline uint2 blend(const uint2 a, const uint2 b, const uint2 mask) { return uint2{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y}; }

  static inline uint extract_0(const uint2 a) { return a.x; } 
  static inline uint extract_1(const uint2 a) { return a.y; } 

  static inline uint2 splat_0(const uint2 a) { return uint2{a.x, a.x}; } 
  static inline uint2 splat_1(const uint2 a) { return uint2{a.y, a.y}; } 

  static inline uint hmin(const uint2 a) { return std::min(a.x, a.y); }
  static inline uint hmax(const uint2 a) { return std::max(a.x, a.y); }

  static inline uint2 shuffle_yx(uint2 a) { return uint2{a.y, a.x}; }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct int2
  {
    inline int2() : x(0), y(0) {}
    inline int2(int a_x, int a_y) : x(a_x), y(a_y) {}
    inline explicit int2(int a_val) : x(a_val), y(a_val) {}
    inline explicit int2(const int a[2]) : x(a[0]), y(a[1]) {}

    inline int& operator[](int i)       { return M[i]; }
    inline int  operator[](int i) const { return M[i]; }

    union
    {
      struct { int x, y; };
      int M[2];
    };
  };

  static inline int2 operator+(const int2 a, const int2 b) { return int2{a.x + b.x, a.y + b.y}; }
  static inline int2 operator-(const int2 a, const int2 b) { return int2{a.x - b.x, a.y - b.y}; }
  static inline int2 operator*(const int2 a, const int2 b) { return int2{a.x * b.x, a.y * b.y}; }
  static inline int2 operator/(const int2 a, const int2 b) { return int2{a.x / b.x, a.y / b.y}; }

  static inline int2 operator * (const int2 a, int b) { return int2{a.x * b, a.y * b}; }
  static inline int2 operator / (const int2 a, int b) { return int2{a.x / b, a.y / b}; }
  static inline int2 operator * (int a, const int2 b) { return int2{a * b.x, a * b.y}; }
  static inline int2 operator / (int a, const int2 b) { return int2{a / b.x, a / b.y}; }

  static inline int2 operator + (const int2 a, int b) { return int2{a.x + b, a.y + b}; }
  static inline int2 operator - (const int2 a, int b) { return int2{a.x - b, a.y - b}; }
  static inline int2 operator + (int a, const int2 b) { return int2{a + b.x, a + b.y}; }
  static inline int2 operator - (int a, const int2 b) { return int2{a - b.x, a - b.y}; }

  static inline int2& operator *= (int2& a, const int2 b) { a.x *= b.x; a.y *= b.y;  return a; }
  static inline int2& operator /= (int2& a, const int2 b) { a.x /= b.x; a.y /= b.y;  return a; }
  static inline int2& operator *= (int2& a, int b) { a.x *= b; a.y *= b;  return a; }
  static inline int2& operator /= (int2& a, int b) { a.x /= b; a.y /= b;  return a; }

  static inline int2& operator += (int2& a, const int2 b) { a.x += b.x; a.y += b.y;  return a; }
  static inline int2& operator -= (int2& a, const int2 b) { a.x -= b.x; a.y -= b.y;  return a; }
  static inline int2& operator += (int2& a, int b) { a.x += b; a.y += b;  return a; }
  static inline int2& operator -= (int2& a, int b) { a.x -= b; a.y -= b;  return a; }

  static inline uint2 operator> (const int2 a, const int2 b) { return uint2{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator< (const int2 a, const int2 b) { return uint2{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator>=(const int2 a, const int2 b) { return uint2{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator<=(const int2 a, const int2 b) { return uint2{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator==(const int2 a, const int2 b) { return uint2{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator!=(const int2 a, const int2 b) { return uint2{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0}; }
 
  static inline int2 operator& (const int2 a, const int2 b) { return int2{a.x & b.x, a.y & b.y}; }
  static inline int2 operator| (const int2 a, const int2 b) { return int2{a.x | b.x, a.y | b.y}; }
  static inline int2 operator~ (const int2 a)                { return int2{~a.x, ~a.y}; }
  static inline int2 operator>>(const int2 a, int b) { return int2{a.x >> b, a.y >> b}; }
  static inline int2 operator<<(const int2 a, int b) { return int2{a.x << b, a.y << b}; }
 
  static inline bool all_of(const int2 a) { return (a.x != 0 && a.y != 0); } 
  static inline bool any_of(const int2 a) { return (a.x != 0 || a.y != 0); } 
 

  static inline void store  (int* p, const int2 a_val) { memcpy(p, &a_val, sizeof(int)*2); }
  static inline void store_u(int* p, const int2 a_val) { memcpy(p, &a_val, sizeof(int)*2); }  


  static inline int2 min  (const int2 a, const int2 b) { return int2{std::min(a.x, b.x), std::min(a.y, b.y)}; }
  static inline int2 max  (const int2 a, const int2 b) { return int2{std::max(a.x, b.x), std::max(a.y, b.y)}; }
  static inline int2 clamp(const int2 u, const int2 a, const int2 b) { return int2{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y)}; }
  static inline int2 clamp(const int2 u, int a, int b) { return int2{clamp(u.x, a, b), clamp(u.y, a, b)}; }

  static inline int2 abs (const int2 a) { return int2{std::abs(a.x), std::abs(a.y)}; } 
  static inline int2 sign(const int2 a) { return int2{sign(a.x), sign(a.y)}; }
  
  static inline  int dot(const int2 a, const int2 b)  { return a.x*b.x + a.y*b.y; }
  static inline  int length(const int2 a) { return std::sqrt(dot(a,a)); }
  static inline  int2 normalize(const int2 a) { int lenInv = int(1)/length(a); return a*lenInv; }

  static inline int2 blend(const int2 a, const int2 b, const uint2 mask) { return int2{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y}; }

  static inline int extract_0(const int2 a) { return a.x; } 
  static inline int extract_1(const int2 a) { return a.y; } 

  static inline int2 splat_0(const int2 a) { return int2{a.x, a.x}; } 
  static inline int2 splat_1(const int2 a) { return int2{a.y, a.y}; } 

  static inline int hmin(const int2 a) { return std::min(a.x, a.y); }
  static inline int hmax(const int2 a) { return std::max(a.x, a.y); }

  static inline int2 shuffle_yx(int2 a) { return int2{a.y, a.x}; }
 

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct float2
  {
    inline float2() : x(0), y(0) {}
    inline float2(float a_x, float a_y) : x(a_x), y(a_y) {}
    inline explicit float2(float a_val) : x(a_val), y(a_val) {}
    inline explicit float2(const float a[2]) : x(a[0]), y(a[1]) {}

    inline float& operator[](int i)       { return M[i]; }
    inline float  operator[](int i) const { return M[i]; }

    union
    {
      struct { float x, y; };
      float M[2];
    };
  };

  static inline float2 operator+(const float2 a, const float2 b) { return float2{a.x + b.x, a.y + b.y}; }
  static inline float2 operator-(const float2 a, const float2 b) { return float2{a.x - b.x, a.y - b.y}; }
  static inline float2 operator*(const float2 a, const float2 b) { return float2{a.x * b.x, a.y * b.y}; }
  static inline float2 operator/(const float2 a, const float2 b) { return float2{a.x / b.x, a.y / b.y}; }

  static inline float2 operator * (const float2 a, float b) { return float2{a.x * b, a.y * b}; }
  static inline float2 operator / (const float2 a, float b) { return float2{a.x / b, a.y / b}; }
  static inline float2 operator * (float a, const float2 b) { return float2{a * b.x, a * b.y}; }
  static inline float2 operator / (float a, const float2 b) { return float2{a / b.x, a / b.y}; }

  static inline float2 operator + (const float2 a, float b) { return float2{a.x + b, a.y + b}; }
  static inline float2 operator - (const float2 a, float b) { return float2{a.x - b, a.y - b}; }
  static inline float2 operator + (float a, const float2 b) { return float2{a + b.x, a + b.y}; }
  static inline float2 operator - (float a, const float2 b) { return float2{a - b.x, a - b.y}; }

  static inline float2& operator *= (float2& a, const float2 b) { a.x *= b.x; a.y *= b.y;  return a; }
  static inline float2& operator /= (float2& a, const float2 b) { a.x /= b.x; a.y /= b.y;  return a; }
  static inline float2& operator *= (float2& a, float b) { a.x *= b; a.y *= b;  return a; }
  static inline float2& operator /= (float2& a, float b) { a.x /= b; a.y /= b;  return a; }

  static inline float2& operator += (float2& a, const float2 b) { a.x += b.x; a.y += b.y;  return a; }
  static inline float2& operator -= (float2& a, const float2 b) { a.x -= b.x; a.y -= b.y;  return a; }
  static inline float2& operator += (float2& a, float b) { a.x += b; a.y += b;  return a; }
  static inline float2& operator -= (float2& a, float b) { a.x -= b; a.y -= b;  return a; }

  static inline uint2 operator> (const float2 a, const float2 b) { return uint2{a.x >  b.x ? 0xFFFFFFFF : 0, a.y >  b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator< (const float2 a, const float2 b) { return uint2{a.x <  b.x ? 0xFFFFFFFF : 0, a.y <  b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator>=(const float2 a, const float2 b) { return uint2{a.x >= b.x ? 0xFFFFFFFF : 0, a.y >= b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator<=(const float2 a, const float2 b) { return uint2{a.x <= b.x ? 0xFFFFFFFF : 0, a.y <= b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator==(const float2 a, const float2 b) { return uint2{a.x == b.x ? 0xFFFFFFFF : 0, a.y == b.y ? 0xFFFFFFFF : 0}; }
  static inline uint2 operator!=(const float2 a, const float2 b) { return uint2{a.x != b.x ? 0xFFFFFFFF : 0, a.y != b.y ? 0xFFFFFFFF : 0}; }

  static inline void store  (float* p, const float2 a_val) { memcpy(p, &a_val, sizeof(float)*2); }
  static inline void store_u(float* p, const float2 a_val) { memcpy(p, &a_val, sizeof(float)*2); }  


  static inline float2 min  (const float2 a, const float2 b) { return float2{std::min(a.x, b.x), std::min(a.y, b.y)}; }
  static inline float2 max  (const float2 a, const float2 b) { return float2{std::max(a.x, b.x), std::max(a.y, b.y)}; }
  static inline float2 clamp(const float2 u, const float2 a, const float2 b) { return float2{clamp(u.x, a.x, b.x), clamp(u.y, a.y, b.y)}; }
  static inline float2 clamp(const float2 u, float a, float b) { return float2{clamp(u.x, a, b), clamp(u.y, a, b)}; }

  static inline float2 abs (const float2 a) { return float2{std::abs(a.x), std::abs(a.y)}; } 
  static inline float2 sign(const float2 a) { return float2{sign(a.x), sign(a.y)}; }

  static inline float2 lerp(const float2 a, const float2 b, float t) { return a + t * (b - a); }
  static inline float2 mix (const float2 a, const float2 b, float t) { return a + t * (b - a); }
  static inline float2 floor(const float2 a)                { return float2{std::floor(a.x), std::floor(a.y)}; }
  static inline float2 ceil(const float2 a)                 { return float2{std::ceil(a.x), std::ceil(a.y)}; }
  static inline float2 rcp (const float2 a)                 { return float2{1.0f/a.x, 1.0f/a.y}; }
  static inline float2 mod (const float2 x, const float2 y) { return x - y * floor(x/y); }
  static inline float2 fract(const float2 x)                { return x - floor(x); }
  static inline float2 sqrt(const float2 a)                 { return float2{std::sqrt(a.x), std::sqrt(a.y)}; }
  static inline float2 inversesqrt(const float2 a)          { return 1.0f/sqrt(a); }
  
  static inline  float dot(const float2 a, const float2 b)  { return a.x*b.x + a.y*b.y; }
  static inline  float length(const float2 a) { return std::sqrt(dot(a,a)); }
  static inline  float2 normalize(const float2 a) { float lenInv = float(1)/length(a); return a*lenInv; }

  static inline float2 blend(const float2 a, const float2 b, const uint2 mask) { return float2{(mask.x == 0) ? b.x : a.x, (mask.y == 0) ? b.y : a.y}; }

  static inline float extract_0(const float2 a) { return a.x; } 
  static inline float extract_1(const float2 a) { return a.y; } 

  static inline float2 splat_0(const float2 a) { return float2{a.x, a.x}; } 
  static inline float2 splat_1(const float2 a) { return float2{a.y, a.y}; } 

  static inline float hmin(const float2 a) { return std::min(a.x, a.y); }
  static inline float hmax(const float2 a) { return std::max(a.x, a.y); }

  static inline float2 shuffle_yx(float2 a) { return float2{a.y, a.x}; }
 

  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static inline float4 to_float32(const uint4 a) { return float4 {float(a.x), float(a.y), float(a.z), float(a.w)}; }
  static inline float4 as_float32(const uint4 a) { float4 res; memcpy(&res, &a, sizeof(uint)*4); return res; }
 

  static inline float4 to_float32(const int4 a) { return float4 {float(a.x), float(a.y), float(a.z), float(a.w)}; }
  static inline float4 as_float32(const int4 a) { float4 res; memcpy(&res, &a, sizeof(uint)*4); return res; }
 

  static inline int4  to_int32 (const float4 a) { return int4 {int (a.x), int (a.y), int (a.z), int (a.w)}; }
  static inline uint4 to_uint32(const float4 a) { return uint4{uint(a.x), uint(a.y), uint(a.z), uint(a.w)}; }
  static inline int4  as_int32 (const float4 a) { int4  res; memcpy(&res, &a, sizeof(int)*4);  return res; }
  static inline uint4 as_uint32(const float4 a) { uint4 res; memcpy(&res, &a, sizeof(uint)*4); return res; } 

  static inline float4 reflect(const float4 dir, const float4 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
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
 

  static inline float3 to_float32(const uint3 a) { return float3 {float(a.x), float(a.y), float(a.z)}; }
  static inline float3 as_float32(const uint3 a) { float3 res; memcpy(&res, &a, sizeof(uint)*3); return res; }
 

  static inline float3 to_float32(const int3 a) { return float3 {float(a.x), float(a.y), float(a.z)}; }
  static inline float3 as_float32(const int3 a) { float3 res; memcpy(&res, &a, sizeof(uint)*3); return res; }
 

  static inline int3  to_int32 (const float3 a) { return int3 {int (a.x), int (a.y), int (a.z)}; }
  static inline uint3 to_uint32(const float3 a) { return uint3{uint(a.x), uint(a.y), uint(a.z)}; }
  static inline int3  as_int32 (const float3 a) { int3  res; memcpy(&res, &a, sizeof(int)*3);  return res; }
  static inline uint3 as_uint32(const float3 a) { uint3 res; memcpy(&res, &a, sizeof(uint)*3); return res; } 

  static inline float3 reflect(const float3 dir, const float3 normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
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
 

  static inline float2 to_float32(const uint2 a) { return float2 {float(a.x), float(a.y)}; }
  static inline float2 as_float32(const uint2 a) { float2 res; memcpy(&res, &a, sizeof(uint)*2); return res; }
 

  static inline float2 to_float32(const int2 a) { return float2 {float(a.x), float(a.y)}; }
  static inline float2 as_float32(const int2 a) { float2 res; memcpy(&res, &a, sizeof(uint)*2); return res; }
 

  static inline int2  to_int32 (const float2 a) { return int2 {int (a.x), int (a.y)}; }
  static inline uint2 to_uint32(const float2 a) { return uint2{uint(a.x), uint(a.y)}; }
  static inline int2  as_int32 (const float2 a) { int2  res; memcpy(&res, &a, sizeof(int)*2);  return res; }
  static inline uint2 as_uint32(const float2 a) { uint2 res; memcpy(&res, &a, sizeof(uint)*2); return res; } 

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
 
  static inline uint4 make_uint4(uint x, uint y, uint z, uint w) { return uint4{x, y, z, w}; }
  static inline int4 make_int4(int x, int y, int z, int w) { return int4{x, y, z, w}; }
  static inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
  static inline uint3 make_uint3(uint x, uint y, uint z) { return uint3{x, y, z}; }
  static inline int3 make_int3(int x, int y, int z) { return int3{x, y, z}; }
  static inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
  static inline uint2 make_uint2(uint x, uint y) { return uint2{x, y}; }
  static inline int2 make_int2(int x, int y) { return int2{x, y}; }
  static inline float2 make_float2(float x, float y) { return float2{x, y}; }
  static inline float3 to_float3(float4 f4)         { return float3(f4.x, f4.y, f4.z); }
  static inline float4 to_float4(float3 v, float w) { return float4(v.x, v.y, v.z, w); }
  static inline uint3  to_uint3 (uint4 f4)          { return uint3(f4.x, f4.y, f4.z);  }
  static inline uint4  to_uint4 (uint3 v, uint w)   { return uint4(v.x, v.y, v.z, w);  }
  static inline int3   to_int3  (int4 f4)           { return int3(f4.x, f4.y, f4.z);   }
  static inline int4   to_int4  (int3 v, int w)     { return int4(v.x, v.y, v.z, w);   }


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct ushort4
  {
    inline ushort4() : x(0), y(0), z(0), w(0) {}
    inline ushort4(ushort a_x, ushort a_y, ushort a_z, ushort a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit ushort4(ushort a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit ushort4(const ushort a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline ushort& operator[](int i)       { return M[i]; }
    inline ushort  operator[](int i) const { return M[i]; }

    union
    {
      struct { ushort x, y, z, w; };
      ushort   M[4];
    };
  };

  struct ushort2
  {
    inline ushort2() : x(0), y(0) {}
    inline ushort2(ushort a_x, ushort a_y) : x(a_x), y(a_y) {}
    inline explicit ushort2(ushort a_val) : x(a_val), y(a_val){}
    inline explicit ushort2(const ushort a[2]) : x(a[0]), y(a[1]) {}

    inline ushort& operator[](int i)       { return M[i]; }
    inline ushort  operator[](int i) const { return M[i]; }

    union
    {
      struct { ushort x, y; };
      ushort   M[2];
      uint64_t u64;
    };
  };

  struct uchar4
  {
    inline uchar4() : x(0), y(0), z(0), w(0) {}
    inline uchar4(uchar a_x, uchar a_y, uchar a_z, uchar a_w) : x(a_x), y(a_y), z(a_z), w(a_w) {}
    inline explicit uchar4(uchar a_val) : x(a_val), y(a_val), z(a_val), w(a_val) {}
    inline explicit uchar4(const uchar a[4]) : x(a[0]), y(a[1]), z(a[2]), w(a[3]) {}

    inline uchar& operator[](int i)       { return M[i]; }
    inline uchar  operator[](int i) const { return M[i]; }

    union
    {
      struct { uchar x, y, z, w; };
      uchar M[4];
      uint32_t u32;
    };
  };


  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  static inline void mat4_rowmajor_mul_mat4(float* __restrict M, const float* __restrict A, const float* __restrict B) // modern gcc compiler succesfuly vectorize such implementation!
  {
  	M[ 0] = A[ 0] * B[ 0] + A[ 1] * B[ 4] + A[ 2] * B[ 8] + A[ 3] * B[12];
  	M[ 1] = A[ 0] * B[ 1] + A[ 1] * B[ 5] + A[ 2] * B[ 9] + A[ 3] * B[13];
  	M[ 2] = A[ 0] * B[ 2] + A[ 1] * B[ 6] + A[ 2] * B[10] + A[ 3] * B[14];
  	M[ 3] = A[ 0] * B[ 3] + A[ 1] * B[ 7] + A[ 2] * B[11] + A[ 3] * B[15];
  	M[ 4] = A[ 4] * B[ 0] + A[ 5] * B[ 4] + A[ 6] * B[ 8] + A[ 7] * B[12];
  	M[ 5] = A[ 4] * B[ 1] + A[ 5] * B[ 5] + A[ 6] * B[ 9] + A[ 7] * B[13];
  	M[ 6] = A[ 4] * B[ 2] + A[ 5] * B[ 6] + A[ 6] * B[10] + A[ 7] * B[14];
  	M[ 7] = A[ 4] * B[ 3] + A[ 5] * B[ 7] + A[ 6] * B[11] + A[ 7] * B[15];
  	M[ 8] = A[ 8] * B[ 0] + A[ 9] * B[ 4] + A[10] * B[ 8] + A[11] * B[12];
  	M[ 9] = A[ 8] * B[ 1] + A[ 9] * B[ 5] + A[10] * B[ 9] + A[11] * B[13];
  	M[10] = A[ 8] * B[ 2] + A[ 9] * B[ 6] + A[10] * B[10] + A[11] * B[14];
  	M[11] = A[ 8] * B[ 3] + A[ 9] * B[ 7] + A[10] * B[11] + A[11] * B[15];
  	M[12] = A[12] * B[ 0] + A[13] * B[ 4] + A[14] * B[ 8] + A[15] * B[12];
  	M[13] = A[12] * B[ 1] + A[13] * B[ 5] + A[14] * B[ 9] + A[15] * B[13];
  	M[14] = A[12] * B[ 2] + A[13] * B[ 6] + A[14] * B[10] + A[15] * B[14];
  	M[15] = A[12] * B[ 3] + A[13] * B[ 7] + A[14] * B[11] + A[15] * B[15];
  }

  static inline void mat4_colmajor_mul_vec4(float* __restrict RES, const float* __restrict B, const float* __restrict V) // modern gcc compiler succesfuly vectorize such implementation!
  {
  	RES[0] = V[0] * B[0] + V[1] * B[4] + V[2] * B[ 8] + V[3] * B[12];
  	RES[1] = V[0] * B[1] + V[1] * B[5] + V[2] * B[ 9] + V[3] * B[13];
  	RES[2] = V[0] * B[2] + V[1] * B[6] + V[2] * B[10] + V[3] * B[14];
  	RES[3] = V[0] * B[3] + V[1] * B[7] + V[2] * B[11] + V[3] * B[15];
  }

  static inline void transpose4(const float4 in_rows[4], float4 out_rows[4])
  {
    CVEX_ALIGNED(16) float rows[16];
    store(rows+0,  in_rows[0]);
    store(rows+4,  in_rows[1]);
    store(rows+8,  in_rows[2]);
    store(rows+12, in_rows[3]);
  
    out_rows[0] = float4{rows[0], rows[4], rows[8],  rows[12]};
    out_rows[1] = float4{rows[1], rows[5], rows[9],  rows[13]};
    out_rows[2] = float4{rows[2], rows[6], rows[10], rows[14]};
    out_rows[3] = float4{rows[3], rows[7], rows[11], rows[15]};
  }

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
      // transpose will change multiplication order (due to we use column major uactually!)
      //
      float4x4 res;
      mat4_rowmajor_mul_mat4((float*)res.m_col, (const float*)rhs.m_col, (const float*)m_col); 
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

    float4 m_col[4];
  };

  static inline float4 operator*(const float4x4& m, const float4& v)
  {
    float4 res;
    mat4_colmajor_mul_vec4((float*)&res, (const float*)&m, (const float*)&v);
    return res;
  }

  static inline float3 operator*(const float4x4& m, const float3& v)
  {
    float4 v2 = float4{v.x, v.y, v.z, 1.0f}; 
    float4 res;                             
    mat4_colmajor_mul_vec4((float*)&res, (const float*)&m, (const float*)&v2);
    return to_float3(res);
  }

  static inline float4x4 transpose(const float4x4& rhs)
  {
    float4x4 res;
    transpose4(rhs.m_col, res.m_col);
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

  static inline float4x4 operator*(float4x4 m1, float4x4 m2)
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

  ///////////////////////////////////////////////////////////////////
  ///// Auxilary functions which are not in the core of library /////
  ///////////////////////////////////////////////////////////////////
  
  inline static uint color_pack_rgba(const float4 rel_col)
  {
    static const float4 const_255(255.0f);
    const uint4 rgba = to_uint32(rel_col*const_255);
    return (rgba[3] << 24) | (rgba[2] << 16) | (rgba[1] << 8) | rgba[0];
  }

  inline static uint color_pack_bgra(const float4 rel_col)
  {
    static const float4 const_255(255.0f);
    const uint4 rgba = to_uint32(shuffle_zyxw(rel_col)*const_255);
    return (rgba[3] << 24) | (rgba[2] << 16) | (rgba[1] << 8) | rgba[0];
  }
  
  inline static float4 color_unpack_bgra(int packedColor)
  {
    const int red   = (packedColor & 0x00FF0000) >> 16;
    const int green = (packedColor & 0x0000FF00) >> 8;
    const int blue  = (packedColor & 0x000000FF) >> 0;
    const int alpha = (packedColor & 0xFF000000) >> 24;
    return float4((float)red, (float)green, (float)blue, (float)alpha)*(1.0f / 255.0f);
  }

  inline static float4 color_unpack_rgba(int packedColor)
  {
    const int blue  = (packedColor & 0x00FF0000) >> 16;
    const int green = (packedColor & 0x0000FF00) >> 8;
    const int red   = (packedColor & 0x000000FF) >> 0;
    const int alpha = (packedColor & 0xFF000000) >> 24;
    return float4((float)red, (float)green, (float)blue, (float)alpha)*(1.0f / 255.0f);
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  // Look At matrix creation
  // return the inverse view matrix
  //
  static inline float4x4 lookAt(float3 eye, float3 center, float3 up)
  {
    float3 x, y, z; // basis; will make a rotation matrix
    z.x = eye.x - center.x;
    z.y = eye.y - center.y;
    z.z = eye.z - center.z;
    z = normalize(z);
    y.x = up.x;
    y.y = up.y;
    y.z = up.z;
    x = cross(y, z); // X vector = Y cross Z
    y = cross(z, x); // Recompute Y = Z cross X
    // cross product gives area of parallelogram, which is < 1.0 for
    // non-perpendicular unit-length vectors; so normalize x, y here
    x = normalize(x);
    y = normalize(y);
    float4x4 M;
    M.set_col(0, float4{ x.x, y.x, z.x, 0.0f });
    M.set_col(1, float4{ x.y, y.y, z.y, 0.0f });
    M.set_col(2, float4{ x.z, y.z, z.z, 0.0f });
    M.set_col(3, float4{ -x.x * eye.x - x.y * eye.y - x.z*eye.z,
                         -y.x * eye.x - y.y * eye.y - y.z*eye.z,
                         -z.x * eye.x - z.y * eye.y - z.z*eye.z,
                         1.0f });
    return M;
  }
  
  static inline float4 packFloatW(const float4& a, float data) { return blend(a, float4(data),            uint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline float4 packIntW(const float4& a, int data)     { return blend(a, as_float32(int4(data)),  uint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline float4 packUIntW(const float4& a, uint data)   { return blend(a, as_float32(uint4(data)), uint4{0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0}); }
  static inline int    extractIntW(const float4& a)            { return as_int(a.w);  }
  static inline uint   extractUIntW(const float4& a)           { return as_uint(a.w); }

  /////////////////////////////////////////
  /////////////// Boxes stuff /////////////
  /////////////////////////////////////////
  
  struct Box4f 
  { 
    inline Box4f()
    {
      boxMin = float4( +std::numeric_limits<float>::infinity() );
      boxMax = float4( -std::numeric_limits<float>::infinity() );   
    }
  
    inline Box4f(const float4 a_bMin, const float4 a_bMax)
    {
      boxMin = a_bMin;
      boxMax = a_bMax;   
    }
  
    inline void include(const LiteMath::float4 p) // please note that this function may override Start/Count pair, so use it carefully
    {                                           
      boxMin = LiteMath::min(boxMin, p);
      boxMax = LiteMath::max(boxMax, p);
    }
  
    inline void include(const Box4f& b) // please note that this function may override Start/Count pair, so use it carefully
    {                                     
      boxMin = LiteMath::min(boxMin, b.boxMin);
      boxMax = LiteMath::max(boxMax, b.boxMax);
    } 
  
    inline void intersect(const Box4f& a_box) 
    {
      boxMin = LiteMath::max(boxMin, a_box.boxMin);
      boxMax = LiteMath::min(boxMax, a_box.boxMax);
    }
  
    inline float surfaceArea() const
    {
      const float4 abc = boxMax - boxMin;
      return 2.0f*(abc[0]*abc[1] + abc[0]*abc[2] + abc[1]*abc[2]);
    }
  
    inline float volume() const 
    {
      const float4 abc = boxMax - boxMin;
      return abc[0]*abc[1]*abc[2];       // #TODO: hmul3
    }
  
    inline void setStart(uint i) { boxMin = packUIntW(boxMin, uint(i)); }
    inline void setCount(uint i) { boxMax = packUIntW(boxMax, uint(i)); }
    inline uint getStart() const { return extractUIntW(boxMin); }
    inline uint getCount() const { return extractUIntW(boxMax); }
    inline bool isAxisAligned(int axis, float split) const { return (boxMin[axis] == boxMax[axis]) && (boxMin[axis]==split); }
  
    float4 boxMin; // as_int(boxMin4f.w) may store index of the object inside the box (or start index of the object sequence)
    float4 boxMax; // as_int(boxMax4f.w) may store size (count) of objects inside the box
  };
  
  struct Ray4f 
  {
    inline Ray4f(){}
    inline Ray4f(const float4& pos, const float4& dir) : posAndNear(pos), dirAndFar(dir) { }
    inline Ray4f(const float4& pos, const float4& dir, float tNear, float tFar) : posAndNear(pos), dirAndFar(dir) 
    { 
      posAndNear = packFloatW(posAndNear, tNear);
      dirAndFar  = packFloatW(dirAndFar,  tFar);
    }
  
    inline Ray4f(const float3& pos, const float3& dir, float tNear, float tFar) : posAndNear(to_float4(pos,tNear)), dirAndFar(to_float4(dir, tFar)) { }
  
    inline float getNear() const { return extract_3(posAndNear); }
    inline float getFar()  const { return extract_3(dirAndFar); }

    inline void  setNear(float tNear) { posAndNear = packFloatW(posAndNear, tNear); } 
    inline void  setFar (float tFar)  { dirAndFar  = packFloatW(dirAndFar,  tFar); } 
  
    float4 posAndNear;
    float4 dirAndFar;
  };
  
  /////////////////////////////////////////
  /////////////// rays stuff //////////////
  /////////////////////////////////////////
  
  /**
  \brief Computes near and far intersection of ray and box
  \param  rayPos     - input ray origin
  \param  rayDirInv  - input inverse ray dir (1.0f/rayDirection)
  \param  boxMin     - input box min
  \param  boxMax     - input box max
  \return (tnear, tfar); if tnear > tfar, no interection is found. 
  */
  static inline float2 Ray4fBox4fIntersection(float4 rayPos, float4 rayDirInv, float4 boxMin, float4 boxMax)
  {
    const float4 lo   = rayDirInv*(boxMin - rayPos);
    const float4 hi   = rayDirInv*(boxMax - rayPos);
    const float4 vmin = min(lo, hi);
    const float4 vmax = max(lo, hi);
    return float2(hmax3(vmin), hmin3(vmax));
  }
    
  /**
  \brief Create eye ray for target x,y and Proj matrix
  \param  x - input x coordinate of pixel
  \param  y - input y coordinate of pixel
  \param  w - input framebuffer image width  
  \param  h - input framebuffer image height
  \param  a_mProjInv - input inverse projection matrix
  \return Eye ray direction; the fourth component will contain +INF as tfar according to Ray4f tnear/tfar storage agreement 
  */
  static inline float4 EyeRayDir4f(float x, float y, float w, float h, float4x4 a_mProjInv) // g_mViewProjInv
  {
    float4 pos = float4( 2.0f * (x + 0.5f) / w - 1.0f, 
                        -2.0f * (y + 0.5f) / h + 1.0f, 
                         0.0f, 
                         1.0f );
  
    pos = a_mProjInv*pos;
    pos = pos/pos.w;
    pos.y *= (-1.0f);      // TODO: do we need remove this (???)
    pos = normalize3(pos);
    pos.w = INF_POSITIVE;
    return pos;
  }
  
  /**
  \brief  calculate overlapping area (volume) of 2 bounding boxes and return it if form of bounding box
  \param  box1 - input first box 
  \param  box2 - input second box
  \return overlaped volume bounding box. If no intersection found, return zero sized bounding box 
  */
  inline Box4f BoxBoxOverlap(const Box4f& box1, const Box4f& box2)
  {
    Box4f tempBox1 = box1;
    Box4f tempBox2 = box2;
    float4 res_min;
    float4 res_max;
    
    for(int axis = 0; axis < 3; ++axis){ // #TODO: unroll loop and vectorize code
      // sort boxes by min
      if(tempBox2.boxMin[axis] < tempBox1.boxMin[axis]){
        float tempSwap = tempBox1.boxMin[axis];
        tempBox1.boxMin[axis] = tempBox2.boxMin[axis];
        tempBox2.boxMin[axis] = tempSwap;
  
        tempSwap = tempBox1.boxMax[axis];
        tempBox1.boxMax[axis] = tempBox2.boxMax[axis];
        tempBox2.boxMax[axis] = tempSwap;
      }
      // check the intersection
      if(tempBox1.boxMax[axis] < tempBox2.boxMin[axis])
        return Box4f(box1.boxMax, box1.boxMax);
  
      // Intersected box
      res_min[axis] = tempBox2.boxMin[axis];
      res_max[axis] = std::min(tempBox1.boxMax[axis], tempBox2.boxMax[axis]);
    }
    return Box4f(res_min, res_max);
  }

};

