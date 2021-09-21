#ifndef TEST_NOISE_H
#define TEST_NOISE_H

#include "LiteMath.h"

#include "noise.h"


using namespace LiteMath;

static inline float fitRange(float x, float src_a, float src_b, float dest_a, float dest_b)
{
  x = x > src_b ? src_b : x;
  x = x < src_a ? src_a : x;
  float range = src_b - src_a;
  float tmp = (x - src_a) / range;

  float range2 = dest_b - dest_a;

  return tmp * range2 + dest_a;
}

static inline float clampf(float x, float minval, float maxval)
{
  return max(min(x,maxval),minval);
}

static inline int clampi(int x, int minval, int maxval)
{
  return max(min(x,maxval),minval);
}

static inline float3 abs3(float3 a)
{
  return make_float3(fabs(a.x), fabs(a.y), fabs(a.z));
}

static inline float4 abs4(float4 a)
{
  return make_float4(fabs(a.x), fabs(a.y), fabs(a.z), fabs(a.w));
}

static inline float3 floor3(float3 v)
{
  return make_float3(floor(v.x), floor(v.y), floor(v.z));
}

static inline float4 floor4(float4 v)
{
  return make_float4(floor(v.x), floor(v.y), floor(v.z), floor(v.w));
}


static inline float3 mod289f3(float3 x)
{
  return x - floor3(x * (1.0 / 289.0)) * 289.0;
}

static inline float4 mod289f4(float4 x)
{
  return x - floor4(x * (1.0 / 289.0)) * 289.0;
}

static inline float4 permute(float4 x)
{
  return mod289f4(((x*34.0) + 1.0)*x);
}

static inline float4 taylorInvSqrt(float4 r)
{
  return 1.79284291400159 - 0.85373472095314 * r;
}

static inline float3 fade(float3 t)
{
  return t*t*t*(t*(t*6.0 - 15.0) + 10.0);
}

static inline float step(float edge, float x)
{
  return x < edge ? 0.0f : 1.0f;
}

static inline float4 step4(float edge, float4 x)
{
  return make_float4(x.x < edge ? 0.0f : 1.0f, x.y < edge ? 0.0f : 1.0f,
                     x.z < edge ? 0.0f : 1.0f, x.w < edge ? 0.0f : 1.0f);
}

static inline float4 step4_(float4 edge, float4 x)
{
  return make_float4(x.x < edge.x ? 0.0f : 1.0f, x.y < edge.y ? 0.0f : 1.0f,
                     x.z < edge.z ? 0.0f : 1.0f, x.w < edge.w ? 0.0f : 1.0f);
}


static inline float rand(float n){return fract(sin(n) * 43758.5453123);}
/*float rand(vec2 co){
    return fract(sin(dot(co.xy ,vec2(12.9898,78.233))) * 43758.5453);
}
*/

// Classic Perlin noise
static inline float cnoise(float3 P)
{
  float3 Pi0 = floor3(P); // Integer part for indexing
  float3 Pi1 = Pi0 + make_float3(1.0, 1.0, 1.0); // Integer part + 1
  Pi0 = mod289f3(Pi0);
  Pi1 = mod289f3(Pi1);
  float3 Pf0 = fract(P); // Fractional part for interpolation
  float3 Pf1 = Pf0 - make_float3(1.0, 1.0, 1.0); // Fractional part - 1.0
  float4 ix = make_float4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  float4 iy = make_float4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  float4 iz0 = make_float4(Pi0.z, Pi0.z, Pi0.z, Pi0.z);
  float4 iz1 = make_float4(Pi1.z, Pi1.z, Pi1.z, Pi1.z);

  float4 ixy = permute(permute(ix) + iy);
  float4 ixy0 = permute(ixy + iz0);
  float4 ixy1 = permute(ixy + iz1);

  float4 gx0 = ixy0 * (1.0 / 7.0);
  float4 gy0 = fract(floor4(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract(gx0);
  float4 gz0 = make_float4(0.5, 0.5, 0.5, 0.5) - abs4(gx0) - abs4(gy0);
  float4 sz0 = step4_(gz0, make_float4(0.0, 0.0, 0.0, 0.0));
  gx0 -= sz0 * (step4(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step4(0.0, gy0) - 0.5);

  float4 gx1 = ixy1 * (1.0 / 7.0);
  float4 gy1 = fract(floor4(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract(gx1);
  float4 gz1 = make_float4(0.5, 0.5, 0.5, 0.5) - abs4(gx1) - abs4(gy1);
  float4 sz1 = step4_(gz1, make_float4(0.0, 0.0, 0.0, 0.0));
  gx1 -= sz1 * (step4(0.0, gx1) - 0.5);
  gy1 -= sz1 * (step4(0.0, gy1) - 0.5);

  float3 g000 = make_float3(gx0.x, gy0.x, gz0.x);
  float3 g100 = make_float3(gx0.y, gy0.y, gz0.y);
  float3 g010 = make_float3(gx0.z, gy0.z, gz0.z);
  float3 g110 = make_float3(gx0.w, gy0.w, gz0.w);
  float3 g001 = make_float3(gx1.x, gy1.x, gz1.x);
  float3 g101 = make_float3(gx1.y, gy1.y, gz1.y);
  float3 g011 = make_float3(gx1.z, gy1.z, gz1.z);
  float3 g111 = make_float3(gx1.w, gy1.w, gz1.w);

  float4 norm0 = taylorInvSqrt(make_float4(dot(g000, g000), dot(g010, g010), dot(g100, g100), dot(g110, g110)));
  g000 *= norm0.x;
  g010 *= norm0.y;
  g100 *= norm0.z;
  g110 *= norm0.w;
  float4 norm1 = taylorInvSqrt(make_float4(dot(g001, g001), dot(g011, g011), dot(g101, g101), dot(g111, g111)));
  g001 *= norm1.x;
  g011 *= norm1.y;
  g101 *= norm1.z;
  g111 *= norm1.w;

  float n000 = dot(g000, Pf0);
  float n100 = dot(g100, make_float3(Pf1.x, Pf0.y, Pf0.z));
  float n010 = dot(g010, make_float3(Pf0.x, Pf1.y, Pf0.z));
  float n110 = dot(g110, make_float3(Pf1.x, Pf1.y, Pf0.z));
  float n001 = dot(g001, make_float3(Pf0.x, Pf0.y, Pf1.z));
  float n101 = dot(g101, make_float3(Pf1.x, Pf0.y, Pf1.z));
  float n011 = dot(g011, make_float3(Pf0.x, Pf1.y, Pf1.z));
  float n111 = dot(g111, Pf1);

  float3 fade_xyz = fade(Pf0);
  float4 n_z  = mix(make_float4(n000, n100, n010, n110), make_float4(n001, n101, n011, n111), fade_xyz.z);
  float2 n_yz = mix(make_float2(n_z.x, n_z.y), make_float2(n_z.z, n_z.w), fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return n_xyz;
}




#endif //TEST_NOISE_H
