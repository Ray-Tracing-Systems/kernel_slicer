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
  typedef unsigned int uint;

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


## for Tests in AllTests
## for Test  in Tests.Tests
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct {{Test.Type}}
  {
    inline {{Test.Type}}() :{% for Coord in Test.XYZW %} {{Coord}}(0){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    inline {{Test.Type}}({% for Coord in Test.XYZW %}{{Test.TypeS}} a_{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}) :{% for Coord in Test.XYZW %} {{Coord}}(a_{{Coord}}){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    inline explicit {{Test.Type}}({{Test.TypeS}} a_val) :{% for Coord in Test.XYZW %} {{Coord}}(a_val){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}
    inline explicit {{Test.Type}}(const {{Test.TypeS}} a[{{Test.VecLen}}]) :{% for Coord in Test.XYZW %} {{Coord}}(a[{{loop.index}}]){% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %} {}

    inline {{Test.TypeS}}& operator[](int i)       { return M[i]; }
    inline {{Test.TypeS}}  operator[](int i) const { return M[i]; }

    union
    {
      struct { {{Test.TypeS}}{% for Coord in Test.XYZW %} {{Coord}}{% if loop.index1 != Test.VecLen %},{% endif %}{% endfor %}; };
      {{Test.TypeS}} M[{{Test.VecLen}}];
    };
  };

  static inline {{Test.Type}} operator+({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} + b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator-({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} - b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator*({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} * b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator/({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} / b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }

  static inline {{Test.Type}} operator * ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} * b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator / ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} / b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator * ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a * b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator / ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a / b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }

  static inline {{Test.Type}} operator + ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} + b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator - ({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a.{{Coord}} - b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator + ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a + b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator - ({{Test.TypeS}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}} {% for Coord in Test.XYZW %}a - b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }

  static inline {{Test.Type}}& operator *= ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} *= b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator /= ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} /= b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator *= ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} *= b; {% endfor %} return a; }
  static inline {{Test.Type}}& operator /= ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} /= b; {% endfor %} return a; }

  static inline {{Test.Type}}& operator += ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} += b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator -= ({{Test.Type}}& a, {{Test.TypeC}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} -= b.{{Coord}}; {% endfor %} return a; }
  static inline {{Test.Type}}& operator += ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} += b; {% endfor %} return a; }
  static inline {{Test.Type}}& operator -= ({{Test.Type}}& a, {{Test.TypeS}} b) { {% for Coord in Test.XYZW %}a.{{Coord}} -= b; {% endfor %} return a; }

  static inline uint{{Test.VecLen}} operator> ({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >  b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator< ({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} <  b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator>=({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >= b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator<=({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >= b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator==({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} == b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} operator!=({{Test.TypeC}} a, {{Test.TypeC}} b) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} != b.{{Coord}} ? 0xFFFFFFFF : 0{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% if not Test.IsFloat %} 
  static inline {{Test.Type}} operator& ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} & b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator| ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} | b.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator~ ({{Test.TypeC}} a)                { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}~a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator>>({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} >> b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} operator<<({{Test.TypeC}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord}} << b{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
 
  static inline bool any_of({{Test.TypeC}} a) { return ({% for Coord in Test.XYZW %}a.{{Coord}} != 0{% if loop.index1 != Test.VecLen %} && {% endif %}{% endfor %}); } 
  static inline bool all_of({{Test.TypeC}} a) { return ({% for Coord in Test.XYZW %}a.{{Coord}} != 0{% if loop.index1 != Test.VecLen %} || {% endif %}{% endfor %}); } 
 
  {% endif %}
  static inline void store  ({{Test.TypeS}}* p, {{Test.TypeC}} a_val) { memcpy(p, &a_val, sizeof({{Test.TypeS}})*{{Test.VecLen}}); }
  static inline void store_u({{Test.TypeS}}* p, {{Test.TypeC}} a_val) { memcpy(p, &a_val, sizeof({{Test.TypeS}})*{{Test.VecLen}}); }  
  
  static inline {{Test.Type}} lerp({{Test.TypeC}} a, {{Test.TypeC}} b, {{Test.TypeS}} t) { return a + t * (b - a); }
  static inline {{Test.Type}} mix ({{Test.TypeC}} a, {{Test.TypeC}} b, {{Test.TypeS}} t) { return a + t * (b - a); }

  static inline {{Test.Type}} min  ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::min(a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} max  ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::max(a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} clamp({{Test.TypeC}} u, {{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}clamp(u.{{Coord}}, a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} clamp({{Test.TypeC}} u, {{Test.TypeS}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}clamp(u.{{Coord}}, a, b){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% if Test.IsSigned %}
  static inline {{Test.Type}} abs ({{Test.TypeC}} a) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::abs(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; } 
  static inline {{Test.Type}} sign({{Test.TypeC}} a) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}sign(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% endif %}{% if Test.IsFloat %}
  static inline {{Test.Type}} floor({{Test.TypeC}} a)                { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::floor(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} ceil({{Test.TypeC}} a)                 { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::ceil(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} rcp ({{Test.TypeC}} a)                 { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}1.0f/a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} mod ({{Test.TypeC}} x, {{Test.TypeC}} y) { return x - y * floor(x/y); }
  static inline {{Test.Type}} fract({{Test.TypeC}} x)                { return x - floor(x); }
  static inline {{Test.Type}} sqrt({{Test.TypeC}} a)                 { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::sqrt(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} inversesqrt({{Test.TypeC}} a)          { return 1.0f/sqrt(a); }
  {% endif %}  
  static inline  {{Test.TypeS}} dot({{Test.TypeC}} a, {{Test.TypeC}} b)  { return {% for Coord in Test.XYZW %}a.{{Coord}}*b.{{Coord}}{% if loop.index1 != Test.VecLen %} + {% endif %}{% endfor %}; }
  static inline  {{Test.TypeS}} length({{Test.TypeC}} a) { return std::sqrt(dot(a,a)); }
  static inline  {{Test.Type}} normalize({{Test.TypeC}} a) { {{Test.TypeS}} lenInv = {{Test.TypeS}}(1)/length(a); return a*lenInv; }
  {% if Test.VecLen == 4 %}
  static inline {{Test.TypeS}}  dot3({{Test.TypeC}} a, {{Test.TypeC}} b)  { return a.x*b.x + a.y*b.y + a.z*b.z; }
  static inline {{Test.TypeS}}  dot4({{Test.TypeC}} a, {{Test.TypeC}} b)  { return dot(a,b); } {% if Test.IsFloat %}
  static inline {{Test.TypeS}}  dot3f({{Test.TypeC}} a, {{Test.TypeC}} b) { return dot3(a,b); }
  static inline {{Test.TypeS}}  dot4f({{Test.TypeC}} a, {{Test.TypeC}} b) { return dot(a,b); }
  static inline {{Test.Type}} dot3v({{Test.TypeC}} a, {{Test.TypeC}} b) { {{Test.TypeS}} res = dot3(a,b); return {{Test.Type}}(res); }
  static inline {{Test.Type}} dot4v({{Test.TypeC}} a, {{Test.TypeC}} b) { {{Test.TypeS}} res = dot(a,b);  return {{Test.Type}}(res); }

  static inline {{Test.TypeS}} length3({{Test.TypeC}} a)  { return std::sqrt(dot3(a,a)); }
  static inline {{Test.TypeS}} length3f({{Test.TypeC}} a) { return std::sqrt(dot3(a,a)); }
  static inline {{Test.TypeS}} length4({{Test.TypeC}} a)  { return std::sqrt(dot4(a,a)); }
  static inline {{Test.TypeS}} length4f({{Test.TypeC}} a) { return std::sqrt(dot4(a,a)); }
  static inline {{Test.Type}} length3v({{Test.TypeC}} a) { {{Test.TypeS}} res = std::sqrt(dot3(a,a)); return {{Test.Type}}(res); }
  static inline {{Test.Type}} length4v({{Test.TypeC}} a) { {{Test.TypeS}} res = std::sqrt(dot4(a,a)); return {{Test.Type}}(res); }
  {% endif %} {% endif %}
  static inline {{Test.Type}} blend({{Test.TypeC}} a, {{Test.TypeC}} b, const uint{{Test.VecLen}} mask) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}(mask.{{Coord}} == 0) ? b.{{Coord}} : a.{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  {% for Coord in Test.XYZW %}
  static inline {{Test.TypeS}} extract_{{loop.index}}({{Test.TypeC}} a) { return a.{{Coord}}; } {% endfor %}
  {% for Coord2 in Test.XYZW %}
  static inline {{Test.Type}} splat_{{loop.index}}({{Test.TypeC}} a) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}a.{{Coord2}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; } {% endfor %}
  {% if Test.VecLen == 4 %}
  static inline {{Test.TypeS}} hmin({{Test.TypeC}} a)  { return std::min(std::min(a.x, a.y), std::min(a.z, a.w) ); }
  static inline {{Test.TypeS}} hmax({{Test.TypeC}} a)  { return std::max(std::max(a.x, a.y), std::max(a.z, a.w) ); }
  static inline {{Test.TypeS}} hmin3({{Test.TypeC}} a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline {{Test.TypeS}} hmax3({{Test.TypeC}} a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline {{Test.Type}} shuffle_xzyw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.x, a.z, a.y, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_yxzw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.x, a.z, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_yzxw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.z, a.x, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_zxyw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.x, a.y, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_zyxw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.y, a.x, a.w{{CLS}}; }
  static inline {{Test.Type}} shuffle_xyxy({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.x, a.y, a.x, a.y{{CLS}}; }
  static inline {{Test.Type}} shuffle_zwzw({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.w, a.z, a.w{{CLS}}; }
  static inline {{Test.Type}} cross3({{Test.TypeC}} a, {{Test.TypeC}} b) 
  {
    const {{Test.Type}} a_yzx = shuffle_yzxw(a);
    const {{Test.Type}} b_yzx = shuffle_yzxw(b);
    return shuffle_yzxw(a*b_yzx - a_yzx*b);
  }
  static inline {{Test.Type}} cross({{Test.TypeC}} a, {{Test.TypeC}} b) { return cross3(a,b); }
  {% else if Test.VecLen == 3 %}
  static inline {{Test.TypeS}} hmin({{Test.TypeC}} a) { return std::min(a.x, std::min(a.y, a.z) ); }
  static inline {{Test.TypeS}} hmax({{Test.TypeC}} a) { return std::max(a.x, std::max(a.y, a.z) ); }

  static inline {{Test.Type}} shuffle_xzy({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.x, a.z, a.y{{CLS}}; }
  static inline {{Test.Type}} shuffle_yxz({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.x, a.z{{CLS}}; }
  static inline {{Test.Type}} shuffle_yzx({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.z, a.x{{CLS}}; }
  static inline {{Test.Type}} shuffle_zxy({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.x, a.y{{CLS}}; }
  static inline {{Test.Type}} shuffle_zyx({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.z, a.y, a.x{{CLS}}; }
  static inline {{Test.Type}} cross({{Test.TypeC}} a, {{Test.TypeC}} b) 
  {
    const {{Test.Type}} a_yzx = shuffle_yzx(a);
    const {{Test.Type}} b_yzx = shuffle_yzx(b);
    return shuffle_yzx(a*b_yzx - a_yzx*b);
  }
  {% else if Test.VecLen == 2 %}
  static inline {{Test.TypeS}} hmin({{Test.TypeC}} a) { return std::min(a.x, a.y); }
  static inline {{Test.TypeS}} hmax({{Test.TypeC}} a) { return std::max(a.x, a.y); }

  static inline {{Test.Type}} shuffle_yx({{Test.Type}} a) { return {{Test.Type}}{{OPN}}a.y, a.x{{CLS}}; }
  {% endif %} 

## endfor
## endfor
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
## for Tests in AllTests
## for Test  in Tests.Tests
  {% if Test.IsFloat %}
  static inline int{{Test.VecLen}}  to_int32 ({{Test.TypeC}} a) { return int{{Test.VecLen}} {{OPN}}{% for Coord in Test.XYZW %}int (a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline uint{{Test.VecLen}} to_uint32({{Test.TypeC}} a) { return uint{{Test.VecLen}}{{OPN}}{% for Coord in Test.XYZW %}uint(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline int{{Test.VecLen}}  as_int32 ({{Test.TypeC}} a) { int{{Test.VecLen}}  res; memcpy(&res, &a, sizeof(int)*{{Test.VecLen}});  return res; }
  static inline uint{{Test.VecLen}} as_uint32({{Test.TypeC}} a) { uint{{Test.VecLen}} res; memcpy(&res, &a, sizeof(uint)*{{Test.VecLen}}); return res; } 

  static inline {{Test.Type}} reflect({{Test.TypeC}} dir, {{Test.TypeC}} normal) { return normal * dot(dir, normal) * (-2.0f) + dir; }
  static inline {{Test.Type}} refract({{Test.TypeC}} incidentVec, {{Test.TypeC}} normal, {{Test.TypeS}} eta)
  {
    {{Test.TypeS}} N_dot_I = dot(normal, incidentVec);
    {{Test.TypeS}} k = {{Test.TypeS}}(1.f) - eta * eta * ({{Test.TypeS}}(1.f) - N_dot_I * N_dot_I);
    if (k < {{Test.TypeS}}(0.f))
      return {{Test.Type}}(0.f);
    else
      return eta * incidentVec - (eta * N_dot_I + std::sqrt(k)) * normal;
  }
  // A floating-point, surface normal vector that is facing the view direction
  static inline {{Test.Type}} faceforward({{Test.TypeC}} N, {{Test.TypeC}} I, {{Test.TypeC}} Ng) { return dot(I, Ng) < {{Test.TypeS}}(0) ? N : {{Test.TypeS}}(-1)*N; }
  {% else %}
  static inline float{{Test.VecLen}} to_float32({{Test.TypeC}} a) { return float{{Test.VecLen}} {{OPN}}{% for Coord in Test.XYZW %}float(a.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline float{{Test.VecLen}} as_float32({{Test.TypeC}} a) { float{{Test.VecLen}} res; memcpy(&res, &a, sizeof(uint)*{{Test.VecLen}}); return res; }
  {% endif %} 
## endfor
## endfor
## for Tests in AllTests
## for Test  in Tests.Tests
  static inline {{Test.Type}} make_{{Test.Type}}({% for Coord in Test.XYZW %}{{Test.TypeS}} {{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}{{Coord}}{% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
## endfor
## endfor
  static inline float3 to_float3(float4 f4)         { return float3(f4.x, f4.y, f4.z); }
  static inline float4 to_float4(float3 v, float w) { return float4(v.x, v.y, v.z, w); }
  static inline uint3  to_uint3 (uint4 f4)          { return uint3(f4.x, f4.y, f4.z);  }
  static inline uint4  to_uint4 (uint3 v, uint w)   { return uint4(v.x, v.y, v.z, w);  }
  static inline int3   to_int3  (int4 f4)           { return int3(f4.x, f4.y, f4.z);   }
  static inline int4   to_int4  (int3 v, int w)     { return int4(v.x, v.y, v.z, w);   }

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

  //private:
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
  
  struct CVEX_ALIGNED(16) Box4f 
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
  
    inline void SetStart(uint i) { boxMin = packUIntW(boxMin, uint(i)); }
    inline void SetCount(uint i) { boxMax = packUIntW(boxMax, uint(i)); }
    inline uint GetStart() const { return extractUIntW(boxMin); }
    inline uint GetCount() const { return extractUIntW(boxMax); }
    inline bool AxisAligned(int axis, float split) const { return (boxMin[axis] == boxMax[axis]) && (boxMin[axis]==split); }
  
    float4 boxMin; // as_int(boxMin4f.w) may store index of the object inside the box (or start index of the object sequence)
    float4 boxMax; // as_int(boxMax4f.w) may store size (count) of objects inside the box
  };
  
  struct CVEX_ALIGNED(16) Ray4f 
  {
    inline Ray4f(){}
    inline Ray4f(const float4& pos, const float4& dir) : posAndNear(pos), dirAndFar(dir) { }
    inline Ray4f(const float4& pos, const float4& dir, float tNear, float tFar) : posAndNear(pos), dirAndFar(dir) 
    { 
      posAndNear = packFloatW(posAndNear, tNear);
      dirAndFar  = packFloatW(dirAndFar,  tFar);
    }
  
    inline Ray4f(const float3& pos, const float3& dir, float tNear, float tFar) : posAndNear(to_float4(pos,tNear)), dirAndFar(to_float4(dir, tFar)) { }
  
    inline float GetNear() const { return extract_3(posAndNear); }
    inline float GetFar()  const { return extract_3(dirAndFar); }
  
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
  
    const float4 vmin = LiteMath::min(lo, hi);
    const float4 vmax = LiteMath::max(lo, hi);
    return float2(hmax3(vmin), hmin3(vmax));
    //return float2(std::max(std::max(vmin[0], vmin[1]), vmin[2]), 
    //              std::min(std::min(vmax[0], vmax[1]), vmax[2]));
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
    pos.y *= (-1.0f); // TODO: remove this (???)
    pos = normalize(pos);
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
