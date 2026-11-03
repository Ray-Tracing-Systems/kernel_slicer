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

#define KGEN_FLAG_RETURN            1
#define KGEN_FLAG_BREAK             2
#define KGEN_FLAG_DONT_SET_EXIT     4
#define KGEN_FLAG_SET_EXIT_NEGATIVE 8

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

complex to_complex(float re) { return make_complex(re, 0.0f);}
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

complex operator+(complex a, complex b) { return make_complex(a.re + b.re, a.im + b.im); }
complex operator+(complex a, float b)   { return make_complex(a.re + b, a.im); }
complex operator+(float a, complex b)   { return make_complex(a + b.re, b.im); }

complex operator-(complex a, complex b) { return make_complex(a.re - b.re, a.im - b.im); }
complex operator-(complex a, float b)   { return make_complex(a.re - b, a.im); }
complex operator-(float a, complex b)   { return make_complex(a - b.re, -b.im); }

complex operator*(complex a, complex b) { return make_complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
complex operator*(complex a, float b)   { return make_complex(a.re * b, a.im * b); }
complex operator*(float a, complex b)   { return make_complex(a * b.re, a * b.im); }

complex operator/(complex a, complex b)
{
  float norm = complex_norm(b);
  return make_complex( (a.re * b.re + a.im * b.im) / norm, (a.im * b.re - a.re * b.im) / norm);
}

complex operator/(complex a, float b)
{
  return make_complex(a.re / b, a.im / b);
}

complex operator/(float a, complex b)
{
  float norm = complex_norm(b);
  return make_complex((a * b.re) / norm, (-a * b.im) / norm);
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
{% if length(HashMaps) > 0 %}

// std::unordered_map<Key,X> ==> buffer of slots {X val; Key key;}, open addressing with linear probing, capacity is power of 2
//
static inline uint kslicer_hmap_hash(uint x) // murmur3 finalizer
{
  x ^= x >> 16; x *= 0x85ebca6bu;
  x ^= x >> 13; x *= 0xc2b2ae35u;
  x ^= x >> 16;
  return x;
}
{% for Map in HashMaps %}

struct {{Map.SlotType}}
{
  {{Map.ValueType}} val;
  {{Map.KeyType}} key;
  {% if Map.PadWords > 0 %}
  uint _pad[{{Map.PadWords}}];
  {% endif %}
};
{% endfor %}
{% endif %}

{% include "inc_ubo.slang" %}

/////////////////////////////////////////////////////////////////////
/////////////////// local functions /////////////////////////////////
/////////////////////////////////////////////////////////////////////

float4x4 make_float4x4_from_cols(float4 a, float4 b, float4 c, float4 d) { return transpose(float4x4(a, b, c, d)); }
float4x4 make_float4x4_from_rows(float4 a, float4 b, float4 c, float4 d) { return float4x4(a, b, c, d); }

float4x4 translate4x4(float3 delta)
{
  return float4x4(float4(1.0f, 0.0f, 0.0f, delta.x),
                  float4(0.0f, 1.0f, 0.0f, delta.y),
                  float4(0.0f, 0.0f, 1.0f, delta.z),
                  float4(0.0f, 0.0f, 0.0f, 1.0f));
}

float4x4 rotate4x4X(float phi)
{
  return float4x4(float4(1.0f, 0.0f,       0.0f,      0.0f),
                  float4(0.0f, +cos(phi), -sin(phi), 0.0f),
                  float4(0.0f, +sin(phi), +cos(phi), 0.0f),
                  float4(0.0f, 0.0f,       0.0f,      1.0f));
}

float4x4 rotate4x4Y(float phi)
{
  return float4x4(float4(+cos(phi), 0.0f, +sin(phi), 0.0f),
                  float4(0.0f,      1.0f, 0.0f,      0.0f),
                  float4(-sin(phi), 0.0f, +cos(phi), 0.0f),
                  float4(0.0f,      0.0f, 0.0f,      1.0f));
}

float4x4 rotate4x4Z(float phi)
{
  return float4x4(float4(+cos(phi), -sin(phi), 0.0f, 0.0f),
                  float4(+sin(phi), +cos(phi), 0.0f, 0.0f),
                  float4(0.0f,      0.0f,      1.0f, 0.0f),
                  float4(0.0f,      0.0f,      0.0f, 1.0f));
}

float3x3 rotate3x3X(float phi)
{
  return float3x3(float3(1.0f, 0.0f,       0.0f),
                  float3(0.0f, +cos(phi), -sin(phi)),
                  float3(0.0f, +sin(phi), +cos(phi)));
}

float3x3 rotate3x3Y(float phi)
{
  return float3x3(float3(+cos(phi), 0.0f, +sin(phi)),
                  float3(0.0f,      1.0f, 0.0f),
                  float3(-sin(phi), 0.0f, +cos(phi)));
}

float3x3 rotate3x3Z(float phi)
{
  return float3x3(float3(+cos(phi), -sin(phi), 0.0f),
                  float3(+sin(phi), +cos(phi), 0.0f),
                  float3(0.0f,      0.0f,      1.0f));
}

float3x3 inverse3x3(float3x3 m)
{
  float det = determinant(m);
  if (abs(det) < 1e-8)
    return float3x3(0, 0, 0, 0, 0, 0, 0, 0, 0);
    
  float invDet = 1.0 / det;
  
  float3x3 adj;
  adj._11 = +(m._22 * m._33 - m._23 * m._32) * invDet;
  adj._12 = -(m._12 * m._33 - m._13 * m._32) * invDet;
  adj._13 = +(m._12 * m._23 - m._13 * m._22) * invDet;
  
  adj._21 = -(m._21 * m._33 - m._23 * m._31) * invDet;
  adj._22 = +(m._11 * m._33 - m._13 * m._31) * invDet;
  adj._23 = -(m._11 * m._23 - m._13 * m._21) * invDet;
  
  adj._31 = +(m._21 * m._32 - m._22 * m._31) * invDet;
  adj._32 = -(m._11 * m._32 - m._12 * m._31) * invDet;
  adj._33 = +(m._11 * m._22 - m._12 * m._21) * invDet;
  
  return adj;
}

//float4x4 inverse4x4(float4x4 m) { return inverse(m); }
//float3 mul4x3(float4x4 m, float3 v) { return (m*float4(v, 1.0f)).xyz; }
//float3 mul3x3(float4x4 m, float3 v) { return (m*float4(v, 0.0f)).xyz; }

float3x3 make_float3x3(float3 a, float3 b, float3 c) { // different way than mat3(a,b,c)
  return float3x3(a.x, a.y, a.z,
                  b.x, b.y, b.z,
                  c.x, c.y, c.z);
}

void set_col2f(inout float2x2 m, int col, float2 value) { m[0][col] = value.x; m[1][col] = value.y; }
void set_col3f(inout float3x3 m, int col, float3 value) { m[0][col] = value.x; m[1][col] = value.y; m[2][col] = value.z; }
void set_col4f(inout float4x4 m, int col, float4 value) { m[0][col] = value.x; m[1][col] = value.y; m[2][col] = value.z; m[3][col] = value.w; }

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

// LiteMath::f32tof16/f16tof32: built-in 'f32tof16/f16tof32' of Slang convert via 'half' in SPIR-V, which needs Float16 and Int16 capabilities
// (shaderFloat16 and shaderInt16 features). 'f32tof16_kgen' uses 32 bit integer operations only and rounds to nearest even as LiteMath on CPU
// (rounding of GLSL.std.450 PackHalf2x16 is implementation defined, AMD rounds to zero); see F. Giesen, 'float_to_half_fast3_rtne'.
// UnpackHalf2x16 is exact and needs only 'Shader' capability.
//
uint f32tof16_kgen(float x)
{
{% if WGPUMode %}
  return f32tof16(x);
{% else %}
  uint       bits = asuint(x);
  const uint sign = bits & 0x80000000u;
  bits ^= sign;
  uint res;
  if(bits >= 0x47800000u)                                // >= 65536: Inf, NaN
    res = (bits > 0x7f800000u) ? 0x7e00u : 0x7c00u;
  else if(bits < 0x38800000u)                            // < 2^-14: subnormal half or zero, addition of 0.5f rounds the mantissa
    res = asuint(asfloat(bits) + 0.5f) - 0x3f000000u;
  else                                                   // normal half: rebias exponent and round to nearest even
    res = (bits + 0xc8000fffu + ((bits >> 13) & 1u)) >> 13;
  return res | (sign >> 16);
{% endif %}
}

float f16tof32_kgen(uint x)
{
{% if WGPUMode %}
  return f16tof32(x);
{% else %}
  const float2 unpacked = spirv_asm { result:$$float2 = OpExtInst glsl450 UnpackHalf2x16 $x };
  return unpacked.x;
{% endif %}
}

{% if AtomicFloatEmul %}
// '-atomicf_emul 1|2' (float atomics for GPUs without shaderBufferFloat32AtomicAdd): float atomic add via CAS on 'uint' view of the same buffer ('a_out_asUint'), declared at the same binding
// by kernel_slicer; 'a_word' is index of 32 bit word in buffer: no pointer casts and no atomics on 'float' in SPIR-V.
// 'a_res' gets the old value, the same as in LiteMath::InterlockedAdd.
//
[ForceInline]
void InterlockedAddEmul1fU(RWStructuredBuffer<uint> pMemU, uint a_word, float data, out float a_res)
{
  uint expected = pMemU[a_word];
  for(;;)
  {
    uint original;
    InterlockedCompareExchange(pMemU[a_word], expected, asuint(asfloat(expected) + data), original);
    if(original == expected)
      break;
    expected = original;
  }
  a_res = asfloat(expected);
}

[ForceInline]
void InterlockedAddEmul1fU(RWStructuredBuffer<uint> pMemU, uint a_word, float data)
{
  float oldVal;
  InterlockedAddEmul1fU(pMemU, a_word, data, oldVal);
}

// LiteMath::InterlockedAdd3f: each channel is atomic separately, not the whole triple.
// 'pMemU' is 'uint' view of the float buffer, declared at the same binding by kernel_slicer ('out_color_asUint'),
// so CAS is ordinary 'InterlockedCompareExchange' on 'uint': no pointer casts and no atomics on 'float' in SPIR-V.
// One loop for all 3 channels: CAS for channels which are not done yet are issued one after another
// without waiting for each other, so their memory latencies overlap (unlike 3 separate CAS loops).
//
[ForceInline]
void InterlockedAdd3f(RWStructuredBuffer<uint> pMemU, int offset, float3 data)
{
  uint3 expected = uint3(pMemU[offset+0], pMemU[offset+1], pMemU[offset+2]);
  bool3 done     = bool3(data.x == 0.0f, data.y == 0.0f, data.z == 0.0f); // adding zero does not change memory
  while(!all(done))
  {
    [unroll]
    for(int k = 0; k < 3; k++)
    {
      if(!done[k])
      {
        uint original;
        InterlockedCompareExchange(pMemU[offset+k], expected[k], asuint(asfloat(expected[k]) + data[k]), original);
        done[k]     = (original == expected[k]);
        expected[k] = original;
      }
    }
  }
}
{% else if not WGPUMode and UseInterlockedAdd3f %}
// LiteMath::InterlockedAdd3f: each channel is atomic separately, not the whole triple (hardware float atomics)
//
[ForceInline]
void InterlockedAdd3f(RWStructuredBuffer<float> pMem, int offset, float3 data)
{
  InterlockedAdd(pMem[offset+0], data.x);
  InterlockedAdd(pMem[offset+1], data.y);
  InterlockedAdd(pMem[offset+2], data.z);
}
{% endif %}
{% if UseCoopVec %}
// LiteNN (apps/LiteMathAux/neural.h): one fully connected layer 'a_out = W*a_in + b' with cooperative vectors (VK_NV_cooperative_vector).
// Weights are fp16, row-major, rows padded to 8 elements; offsets 'a_matOffs' and 'a_biasOffs' are in elements of half.
// Input and output are converted to fp16: the only floating point combination supported by hardware.
//
[ForceInline]
void LiteNN_LinearForward<let IN : int, let OUT : int, let N0 : int, let N1 : int>(RWStructuredBuffer<half> a_weights, uint a_matOffs, uint a_biasOffs,
                                                                                   float a_in[N0], inout float a_out[N1])
{
  CoopVec<half, IN> x;
  [ForceUnroll]
  for(int i = 0; i < IN; i++)
    x[i] = half(a_in[i]);
  const CoopVec<half, OUT> y = coopVecMatMulAdd<half, OUT>(x, CoopVecComponentType::Float16,
                                                           a_weights, a_matOffs*2, CoopVecComponentType::Float16,
                                                           a_weights, a_biasOffs*2, CoopVecComponentType::Float16,
                                                           CoopVecMatrixLayout::RowMajor, false, ((IN + 7)/8)*16);
  [ForceUnroll]
  for(int i = 0; i < OUT; i++)
    a_out[i] = float(y[i]);
}

[ForceInline]
void LiteNN_LinearForward_ReLU<let IN : int, let OUT : int, let N0 : int, let N1 : int>(RWStructuredBuffer<half> a_weights, uint a_matOffs, uint a_biasOffs,
                                                                                        float a_in[N0], inout float a_out[N1])
{
  CoopVec<half, IN> x;
  [ForceUnroll]
  for(int i = 0; i < IN; i++)
    x[i] = half(a_in[i]);
  CoopVec<half, OUT> y = coopVecMatMulAdd<half, OUT>(x, CoopVecComponentType::Float16,
                                                     a_weights, a_matOffs*2, CoopVecComponentType::Float16,
                                                     a_weights, a_biasOffs*2, CoopVecComponentType::Float16,
                                                     CoopVecMatrixLayout::RowMajor, false, ((IN + 7)/8)*16);
  y = max(y, CoopVec<half, OUT>(half(0.0f)));
  [ForceUnroll]
  for(int i = 0; i < OUT; i++)
    a_out[i] = float(y[i]);
}

[ForceInline]
void LiteNN_ReLU<let N : int>(inout float a_data[N], uint a_size)
{
  for(uint i = 0; i < a_size; i++)
    a_data[i] = max(a_data[i], 0.0f);
}
{% endif %}


## for LocalDecl in LocalFuncDecls
{{LocalDecl}};
## endfor

## for LocalFunc in LocalFunctions  
{{LocalFunc}}

## endfor
