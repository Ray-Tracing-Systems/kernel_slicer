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
  {% endif %}
  static inline void store  ({{Test.TypeS}}* p, {{Test.TypeC}} a_val) { memcpy(p, &a_val, sizeof({{Test.TypeS}})*{{Test.VecLen}}); }
  static inline void store_u({{Test.TypeS}}* p, {{Test.TypeC}} a_val) { memcpy(p, &a_val, sizeof({{Test.TypeS}})*{{Test.VecLen}}); }  
  
  static inline {{Test.Type}} lerp({{Test.TypeC}} a, {{Test.TypeC}} b, {{Test.TypeS}} t) { return a + t * (b - a); }
  static inline {{Test.Type}} mix ({{Test.TypeC}} a, {{Test.TypeC}} b, {{Test.TypeS}} t) { return a + t * (b - a); }

  static inline  {{Test.TypeS}} dot({{Test.TypeC}} a, {{Test.TypeC}} b)  { return {% for Coord in Test.XYZW %}a.{{Coord}}*b.{{Coord}}{% if loop.index1 != Test.VecLen %} + {% endif %}{% endfor %}; }

  static inline {{Test.Type}} min  ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::min(a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} max  ({{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}std::max(a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} clamp({{Test.TypeC}} u, {{Test.TypeC}} a, {{Test.TypeC}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}clamp(u.{{Coord}}, a.{{Coord}}, b.{{Coord}}){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  static inline {{Test.Type}} clamp({{Test.TypeC}} u, {{Test.TypeS}} a, {{Test.TypeS}} b) { return {{Test.Type}}{{OPN}}{% for Coord in Test.XYZW %}clamp(u.{{Coord}}, a, b){% if loop.index1 != Test.VecLen %}, {% endif %}{% endfor %}{{CLS}}; }
  

## endfor
## endfor


};
