/////////////////////////////////////////////////////////////////////
/////////////  Required  Shader Features ////////////////////////////
/////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
static const float FLT_MAX = 1e37f;
static const float FLT_MIN = -1e37f;
static const float FLT_EPSILON = 1e-6f;
static const float EPSILON     = 1e-6f;
static const float DEG_TO_RAD  = 0.017453293f; 

float  SQR(float x)  { return x * x; }
double SQR(double x) { return x * x; }
int    SQR(int x)    { return x * x; }
uint   SQR(uint x)   { return x * x; }

bool  isfinite(float x)            { return !isinf(x); }
float copysign(float mag, float s) { return abs(mag)*sign(s); }
{% if UseComplex %}

struct complex
{
  float re, im;
};

complex make_complex(float re, float im) { 
  complex res;
  res.re = re;
  res.im = im;
  return res;
}

complex to_complex(float re)              { return make_complex(re, 0.0f);}
complex complex_add(complex a, complex b) { return make_complex(a.re + b.re, a.im + b.im); }
complex complex_sub(complex a, complex b) { return make_complex(a.re - b.re, a.im - b.im); }
complex complex_mul(complex a, complex b) { return make_complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
complex complex_div(complex a, complex b) {
  const float scale = 1 / (b.re * b.re + b.im * b.im);
  return make_complex(scale * (a.re * b.re + a.im * b.im), scale * (a.im * b.re - a.re * b.im));
}

complex real_add_complex(float value, complex z) { return complex_add(to_complex(value),z); }
complex real_sub_complex(float value, complex z) { return complex_sub(to_complex(value),z); }
complex real_mul_complex(float value, complex z) { return complex_mul(to_complex(value),z); }
complex real_div_complex(float value, complex z) { return complex_div(to_complex(value),z); }

complex complex_add_real(complex z, float value) { return complex_add(z, to_complex(value)); }
complex complex_sub_real(complex z, float value) { return complex_sub(z, to_complex(value)); }
complex complex_mul_real(complex z, float value) { return complex_mul(z, to_complex(value)); }
complex complex_div_real(complex z, float value) { return complex_div(z, to_complex(value)); }

float real(complex z) { return z.re;}
float imag(complex z) { return z.im; }
float complex_norm(complex z) { return z.re * z.re + z.im * z.im; }
float complex_abs(complex z) { return sqrt(complex_norm(z)); }
complex complex_sqrt(complex z) 
{
  float n = complex_abs(z);
  float t1 = sqrt(0.5f * (n + abs(z.re)));
  float t2 = 0.5f * z.im / t1;
  if (n == 0.0f)
    return to_complex(0.0f);
  if (z.re >= 0.0f)
    return make_complex(t1, t2);
  else
    return make_complex(abs(t2), copysign(t1, z.im));
}

{% endif %}
## for Def in Defines  
{{Def}}
## endfor
{% for UserDef in UserTypeDefs %}
#define {{UserDef.Original}} {{UserDef.Redefined}}
{% endfor %}
## for Decl in ClassDecls  
{% if Decl.IsTdef %}
{{Decl.Text}}
{% endif %}
## endfor
## for Decl in ClassDecls  
{% if not Decl.IsTdef %}
{{Decl.Text}}
{% endif %}
## endfor

{% include "inc_ubo.slang" %}

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

float4x4 translate4x4(float3 delta)
{
  return float4x4(float4(1.0, 0.0, 0.0, 0.0),
                  float4(0.0, 1.0, 0.0, 0.0),
                  float4(0.0, 0.0, 1.0, 0.0),
                  float4(delta, 1.0));
}

float4x4 rotate4x4X(float phi)
{
  return float4x4(float4(1.0f, 0.0f,  0.0f,           0.0f),
                  float4(0.0f, +cos(phi),  +sin(phi), 0.0f),
                  float4(0.0f, -sin(phi),  +cos(phi), 0.0f),
                  float4(0.0f, 0.0f,       0.0f,      1.0f));
}

float4x4 rotate4x4Y(float phi)
{
  return float4x4(float4(+cos(phi), 0.0f, -sin(phi), 0.0f),
                  float4(0.0f,      1.0f, 0.0f,      0.0f),
                  float4(+sin(phi), 0.0f, +cos(phi), 0.0f),
                  float4(0.0f,      0.0f, 0.0f,      1.0f));
}

float4x4 rotate4x4Z(float phi)
{
  return float4x4(float4(+cos(phi), sin(phi), 0.0f, 0.0f),
                  float4(-sin(phi), cos(phi), 0.0f, 0.0f),
                  float4(0.0f,      0.0f,     1.0f, 0.0f),
                  float4(0.0f,      0.0f,     0.0f, 1.0f));
}

//float4x4 inverse4x4(float4x4 m) { return inverse(m); }
//float3 mul4x3(float4x4 m, float3 v) { return (m*float4(v, 1.0f)).xyz; }
//float3 mul3x3(float4x4 m, float3 v) { return (m*float4(v, 0.0f)).xyz; }

float3x3 make_float3x3(float3 a, float3 b, float3 c) { // different way than mat3(a,b,c)
  return float3x3(a.x, b.x, c.x,
                  a.y, b.y, c.y,
                  a.z, b.z, c.z);
}

float4 cross3(float4 a, float4 b) { return float4(cross(a.xyz, b.xyz), 1.0f); }

struct Box4f 
{ 
  float4 boxMin; 
  float4 boxMax;
};  

// Have them in lang to do less rewriting, don't have them in GLSL backend
//
uint4  make_uint4(uint x, uint y, uint z, uint w) { return uint4(x, y, z, w); }
int4   make_int4(int x, int y, int z, int w) { return int4(x, y, z, w); }
float4 make_float4(float x, float y, float z, float w) { return float4(x, y, z, w); }
uint3  make_uint3(uint x, uint y, uint z) { return uint3(x, y, z); }
int3   make_int3(int x, int y, int z) { return int3(x, y, z); }
float3 make_float3(float x, float y, float z) { return float3(x, y, z); }
uint2  make_uint2(uint x, uint y) { return uint2(x, y); }
int2   make_int2(int x, int y) { return int2(x, y); }
float2 make_float2(float x, float y) { return float2(x, y); }

float3 to_float3(float4 f4)         { return f4.xyz; }
float4 to_float4(float3 v, float w) { return float4(v.x, v.y, v.z, w); }
uint3  to_uint3 (uint4 f4)          { return f4.xyz;  }
uint4  to_uint4 (uint3 v, uint w)   { return uint4(v.x, v.y, v.z, w);  }
int3   to_int3  (int4 f4)           { return f4.xyz;   }
int4   to_int4  (int3 v, int w)     { return int4(v.x, v.y, v.z, w);   }

float4 mul4x4x4(float4x4 m, float4 v) { return mul(m,v); }
float3 mul3x3  (float4x4 m, float3 v) { return to_float3(mul(m, to_float4(v, 0.0f))); }
float3 mul4x3  (float4x4 m, float3 v) { return to_float3(mul(m, to_float4(v, 1.0f))); }

float4   operator*(float4x4 m, float4 v) { return mul(m,v); }
float3   operator*(float4x4 m, float3 v) { return mul(m, float4(v,1.0f)).xyz; }
float3   operator*(float3x3 m, float3 v) { return mul(m,v); }

double4   operator*(double4x4 m,  double4 v) { return mul(m,v); }
double3   operator*(double4x4 m,  double3 v) { return mul(m, double4(v,1.0f)).xyz; }
double3   operator*(double3x3 m,  double3 v) { return mul(m,v); }

static inline uint bitCount(uint x) { return countbits(x); }

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor
