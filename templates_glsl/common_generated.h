/////////////////////////////////////////////////////////////////////
/////////////  Required  Shader Features ////////////////////////////
/////////////////////////////////////////////////////////////////////
{% if GlobalUseInt8 or DeclsUseInt8 %} 
#extension GL_EXT_shader_explicit_arithmetic_types_int8: require
{% endif %}
{% if GlobalUseInt16 or DeclsUseInt16 %} 
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
{% endif %}
{% if GlobalUseInt64 %} 
#extension GL_EXT_shader_explicit_arithmetic_types_int64: require
{% endif %}
{% if GlobalUseHalf or DeclsUseHalf %} 
#extension GL_EXT_shader_16bit_storage : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16: enable
{% endif %}

/////////////////////////////////////////////////////////////////////
/////////////////// include files ///////////////////////////////////
/////////////////////////////////////////////////////////////////////
## for Incl in Includes  
#include "{{Incl}}"
## endfor

/////////////////////////////////////////////////////////////////////
/////////////////// declarations in class ///////////////////////////
/////////////////////////////////////////////////////////////////////
#ifndef uint32_t
#define uint32_t uint
#endif
#define FLT_MAX 1e37f
#define FLT_MIN -1e37f
#define FLT_EPSILON 1e-6f
#define DEG_TO_RAD  0.017453293f
#define unmasked
#define half  float16_t
#define half2 f16vec2
#define half3 f16vec3
#define half4 f16vec4
bool  isfinite(float x)            { return !isinf(x); }
float copysign(float mag, float s) { return abs(mag)*sign(s); }
uint  f32tof16(float x)             { return packHalf2x16(vec2(x, 0.0f)) & 0xFFFFu; } // LiteMath::f32tof16 (HLSL semantic)
float f16tof32(uint x)              { return unpackHalf2x16(x).x; }                // LiteMath::f16tof32 (HLSL semantic)
{% if AtomicFloatEmul %}

// '-atomicf_emul 1|2' (GPUs without shaderBufferFloat32AtomicAdd): float atomic add via CAS on 'uint' view of the same buffer ('a_out_asUint'),
// declared at the same binding by kernel_slicer; 'a_word' is index of 32 bit word in buffer. GLSL can't pass buffer to a function, so these are macros.
// 'a_res' gets the old value, the same as in LiteMath::InterlockedAdd.
#define InterlockedAddEmul1fURes(a_bufferU, a_word, a_data, a_res) do {                                                             \
  const uint  kgenWord     = (a_word);                                                                                         \
  const float kgenData     = (a_data);                                                                                         \
  uint        kgenExpected = a_bufferU[kgenWord];                                                                              \
  for(;;) {                                                                                                                    \
    const uint kgenOriginal = atomicCompSwap(a_bufferU[kgenWord], kgenExpected, floatBitsToUint(uintBitsToFloat(kgenExpected) + kgenData)); \
    if(kgenOriginal == kgenExpected) break;                                                                                    \
    kgenExpected = kgenOriginal;                                                                                               \
  }                                                                                                                            \
  a_res = uintBitsToFloat(kgenExpected); } while(false)
#define InterlockedAddEmul1fU(a_bufferU, a_word, a_data) do { float kgenOldVal; InterlockedAddEmul1fURes(a_bufferU, a_word, a_data, kgenOldVal); } while(false)
{% endif %}
{% if UseInterlockedAdd3f %}

// LiteMath::InterlockedAdd3f(buffer, offset, data): each channel is atomic separately, not the whole triple.
// GLSL can't pass buffer to a function, so it is a macro; 'do { ... } while(false)' makes it a single statement ('if(c) InterlockedAdd3f(...); else ...').
{% if AtomicFloatEmul %}
#define InterlockedAdd3f(a_bufferU, a_offset, a_data) do {                                                                          \
  const uint kgen3Offs = uint(a_offset);                                                                                       \
  const vec3 kgen3Data = (a_data);                                                                                             \
  if(kgen3Data.x != 0.0f) InterlockedAddEmul1fU(a_bufferU, kgen3Offs + 0, kgen3Data.x);                                        \
  if(kgen3Data.y != 0.0f) InterlockedAddEmul1fU(a_bufferU, kgen3Offs + 1, kgen3Data.y);                                        \
  if(kgen3Data.z != 0.0f) InterlockedAddEmul1fU(a_bufferU, kgen3Offs + 2, kgen3Data.z); } while(false)
{% else %}
#define InterlockedAdd3f(a_buffer, a_offset, a_data) do {                                                                           \
  const int  kgen3Offs = int(a_offset);                                                                                        \
  const vec3 kgen3Data = (a_data);                                                                                             \
  atomicAdd(a_buffer[kgen3Offs + 0], kgen3Data.x);                                                                             \
  atomicAdd(a_buffer[kgen3Offs + 1], kgen3Data.y);                                                                             \
  atomicAdd(a_buffer[kgen3Offs + 2], kgen3Data.z); } while(false)
{% endif %}
{% endif %}
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

{% include "inc_ubo.glsl" %}

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

mat4 make_float4x4_from_cols(vec4 a, vec4 b, vec4 c, vec4 d) { return mat4(a, b, c, d); } // swaped rows/cols logic between LiteMath <=> GLSL
mat4 make_float4x4_from_rows(vec4 a, vec4 b, vec4 c, vec4 d)                              // swaped rows/cols logic between LiteMath <=> GLSL
{
  return mat4(
      a.x, b.x, c.x, d.x,
      a.y, b.y, c.y, d.y,
      a.z, b.z, c.z, d.z,
      a.w, b.w, c.w, d.w
  );
}

mat4 translate4x4(vec3 delta)
{
  return mat4(vec4(1.0, 0.0, 0.0, 0.0),
              vec4(0.0, 1.0, 0.0, 0.0),
              vec4(0.0, 0.0, 1.0, 0.0),
              vec4(delta, 1.0));
}

mat4 rotate4x4X(float phi)
{
  return mat4(vec4(1.0f, 0.0f,  0.0f,           0.0f),
              vec4(0.0f, +cos(phi),  +sin(phi), 0.0f),
              vec4(0.0f, -sin(phi),  +cos(phi), 0.0f),
              vec4(0.0f, 0.0f,       0.0f,      1.0f));
}

mat4 rotate4x4Y(float phi)
{
  return mat4(vec4(+cos(phi), 0.0f, -sin(phi), 0.0f),
              vec4(0.0f,      1.0f, 0.0f,      0.0f),
              vec4(+sin(phi), 0.0f, +cos(phi), 0.0f),
              vec4(0.0f,      0.0f, 0.0f,      1.0f));
}

mat4 rotate4x4Z(float phi)
{
  return mat4(vec4(+cos(phi), sin(phi), 0.0f, 0.0f),
              vec4(-sin(phi), cos(phi), 0.0f, 0.0f),
              vec4(0.0f,      0.0f,     1.0f, 0.0f),
              vec4(0.0f,      0.0f,     0.0f, 1.0f));
}

mat3 rotate3x3X(float phi)
{
  return mat3(vec3(1.0f, 0.0f, 0.0f),
              vec3(0.0f, +cos(phi),  +sin(phi)),
              vec3(0.0f, -sin(phi),  +cos(phi)));
}

mat3 rotate3x3Y(float phi)
{
  return mat3(vec3(+cos(phi), 0.0f, -sin(phi)),
              vec3(0.0f,      1.0f, 0.0f     ),
              vec3(+sin(phi), 0.0f, +cos(phi)));
}

mat3 rotate3x3Z(float phi)
{
  return mat3(vec3(+cos(phi), sin(phi), 0.0f),
              vec3(-sin(phi), cos(phi), 0.0f),
              vec3(0.0f,      0.0f,     1.0f));
}

mat4 inverse4x4(mat4 m)     { return inverse(m); }
vec3 mul4x3(mat4 m, vec3 v) { return (m*vec4(v, 1.0f)).xyz; }
vec3 mul3x3(mat4 m, vec3 v) { return (m*vec4(v, 0.0f)).xyz; }

mat3 make_float3x3(vec3 a, vec3 b, vec3 c) { // different way than mat3(a,b,c)
  return mat3(a.x, b.x, c.x,
              a.y, b.y, c.y,
              a.z, b.z, c.z);
}

void set_row2f(inout mat2 m, int col, vec2 value) { m[0][col] = value.x; m[1][col] = value.y; }
void set_row3f(inout mat3 m, int col, vec3 value) { m[0][col] = value.x; m[1][col] = value.y; m[2][col] = value.z; }
void set_row4f(inout mat4 m, int col, vec4 value) { m[0][col] = value.x; m[1][col] = value.y; m[2][col] = value.z; m[3][col] = value.w; }

vec4 cross3(vec4 a, vec4 b) { return vec4(cross(a.xyz, b.xyz), 1.0f); }

struct Box4f 
{ 
  vec4 boxMin; 
  vec4 boxMax;
};  

#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8
#define KGEN_REDUCTION_LAST_STEP    16
## for Def in Defines  
{{Def}}
## endfor

## for LocalDecl in LocalFuncDecls
{{LocalDecl}};
## endfor

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor