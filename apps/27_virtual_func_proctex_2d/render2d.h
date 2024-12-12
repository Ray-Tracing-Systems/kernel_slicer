#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "LiteMath.h"
using namespace LiteMath;

struct IProcTexture2D
{
  static constexpr uint32_t TAG_EMPTY        = 0;   
  static constexpr uint32_t TAG_YELLOW_NOISE = 1;
  static constexpr uint32_t TAG_MANDELBROT   = 2;
  static constexpr uint32_t TAG_OCEAN        = 3; 
  static constexpr uint32_t TAG_VORONOI      = 4; 
  static constexpr uint32_t TAG_PERLIN       = 5;

  virtual uint32_t GetTag() const { return TAG_EMPTY; }      
  virtual float3 Evaluate(float2 tc) const { return float3(0.0f); }

  uint32_t m_tag = TAG_EMPTY;
};

class ProcRender2D 
{
public:

  ProcRender2D();
  ~ProcRender2D();

  virtual void Fractal(int w, int h, uint32_t* outData [[size("w*h")]], int a_branchMode);

  virtual void CommitDeviceData() {}                                                           // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; } // will be overriden in generated class    

  static constexpr uint32_t TOTAL_IMPLEMANTATIONS = 5;

  enum {BRANCHING_LITE = 0, BRANCHING_MEDIUM = 1, BRANCHING_HEAVY = 2 }; 

protected:

  virtual void kernel2D_EvaluateTextures(int w, int h, uint32_t* outData, int a_branchMode);
  float m_time;

  std::vector<IProcTexture2D*> allProcTextures; 
  void InitAllTextures();
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct Empty2D : public IProcTexture2D
{
  Empty2D() { m_tag = GetTag(); }
  uint32_t GetTag() const override { return TAG_EMPTY; }      
  float3 Evaluate(float2 tc) const override { return float3(0,0,0); }
  uint32_t m_dummy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float2 grad(int2 z )  // replace this anything that returns a random vector
{
  // 2D to 1D  (feel free to replace by some other)
  int n = z.x+z.y*11111;
  // Hugo Elias hash (feel free to replace by another one)
  n = (n<<13)^n;
  n = (n*(n*n*15731+789221)+1376312589)>>16;
  // Perlin style vectors
  n &= 7;
  float2 gr = float2(n&1,n>>1)*2.0-1.0;
  return ( n>=6 ) ? float2(0.0f,gr.x) : 
         ( n>=4 ) ? float2(gr.x,0.0f) : gr;                           
}

static inline float noise(float2 p)
{
  int2   i = int2(floor( p ));
  float2 f =      fract( p );

  float2 u = f*f*(3.0f - 2.0f*f); // feel free to replace by a quintic smoothstep instead
  return mix( mix( dot( grad( i + int2(0,0) ), f - float2(0.0,0.0) ), 
                   dot( grad( i + int2(1,0) ), f - float2(1.0,0.0) ), u.x),
              mix( dot( grad( i + int2(0,1) ), f - float2(0.0,1.0) ), 
                   dot( grad( i + int2(1,1) ), f - float2(1.0,1.0) ), u.x), u.y);
}


struct YellowNoise : public IProcTexture2D
{
  YellowNoise() { m_tag = GetTag(); }  

  uint32_t GetTag()          const override { return TAG_YELLOW_NOISE; }      
  float3 Evaluate(float2 tc) const override 
  {
    float4 uv = 8.0f*float4(tc.x, tc.y, 0.0f, 1.0f);

    float4x4 m = float4x4( 1.6,  1.2, 0.0f,  0.0f, 
                          -1.2,  1.6, 0.0f,  0.0f,
                           0.0f, 0.0f, 1.0f, 0.0f,
                           0.0f, 0.0f, 0.0f, 1.0f);
    float f = 0.0;                       
		f  = 0.5000f*noise( float2(uv.x, uv.y) ); uv = m*uv;
		f += 0.2500f*noise( float2(uv.x, uv.y) ); uv = m*uv;
		f += 0.1250f*noise( float2(uv.x, uv.y) ); uv = m*uv;
		f += 0.0625f*noise( float2(uv.x, uv.y) ); uv = m*uv; 
    f = 0.5f + 0.5*f;
    f *= smoothstep(0.0f, 0.005f, abs(tc.x-0.6f) );	

    return float3(f, f*f, 0.0f); 
  }
  uint32_t m_dummy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline int mandel(float c_re, float c_im, int count) 
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) 
  {
    if (z_re * z_re + z_im * z_im > 4.)
        break;
    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.f * z_re * z_im;
    z_re = c_re + new_re;
    z_im = c_im + new_im;
      
  }
  return i;
}


struct Mandelbrot2D : public IProcTexture2D
{
  Mandelbrot2D() { m_tag = GetTag(); }
  uint32_t GetTag() const override { return TAG_MANDELBROT; }      
  
  float3 Evaluate(float2 tc) const override 
  { 
    const int index = mandel((tc.x-0.5f)*1.25f, tc.y*1.25f, 100);

    const int r1 = std::min((index*128)/32, 255);
    const int g1 = std::min((index*128)/25, 255);
    const int b1 = std::min((index*index), 255);

    const float fr1 = float(r1)/255.0f;
    const float fg1 = float(g1)/255.0f;
    const float fb1 = float(b1)/255.0f;

    return float3(fr1, fg1, fb1); 
  }

  uint32_t m_dummy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// https://www.shadertoy.com/view/3tKBDz

static inline float circ(float2 pos, float2 c, float s)
{
  c = abs(pos - c);
  c = min(c, 1.0f - c);

  return smoothstep(0.0f, 0.002f, std::sqrt(s) - std::sqrt(dot(c, c))) * -1.0f;
}

static inline float waterlayer(float2 tc)
{
  float2 uv = float2(mod(tc.x, 1.0f), mod(tc.y, 1.0f));
  float ret = 1.0;
  ret += circ(uv, float2(0.37378, 0.277169), 0.0268181);
  ret += circ(uv, float2(0.0317477, 0.540372), 0.0193742);
  ret += circ(uv, float2(0.430044, 0.882218), 0.0232337);
  ret += circ(uv, float2(0.641033, 0.695106), 0.0117864);
  ret += circ(uv, float2(0.0146398, 0.0791346), 0.0299458);
  ret += circ(uv, float2(0.43871, 0.394445), 0.0289087);
  ret += circ(uv, float2(0.909446, 0.878141), 0.028466);
  ret += circ(uv, float2(0.310149, 0.686637), 0.0128496);
  ret += circ(uv, float2(0.928617, 0.195986), 0.0152041);
  ret += circ(uv, float2(0.0438506, 0.868153), 0.0268601);
  ret += circ(uv, float2(0.308619, 0.194937), 0.00806102);
  ret += circ(uv, float2(0.349922, 0.449714), 0.00928667);
  ret += circ(uv, float2(0.0449556, 0.953415), 0.023126);
  ret += circ(uv, float2(0.117761, 0.503309), 0.0151272);
  ret += circ(uv, float2(0.563517, 0.244991), 0.0292322);
  ret += circ(uv, float2(0.566936, 0.954457), 0.00981141);
  ret += circ(uv, float2(0.0489944, 0.200931), 0.0178746);
  ret += circ(uv, float2(0.569297, 0.624893), 0.0132408);
  ret += circ(uv, float2(0.298347, 0.710972), 0.0114426);
  ret += circ(uv, float2(0.878141, 0.771279), 0.00322719);
  ret += circ(uv, float2(0.150995, 0.376221), 0.00216157);
  ret += circ(uv, float2(0.119673, 0.541984), 0.0124621);
  ret += circ(uv, float2(0.629598, 0.295629), 0.0198736);
  ret += circ(uv, float2(0.334357, 0.266278), 0.0187145);
  ret += circ(uv, float2(0.918044, 0.968163), 0.0182928);
  ret += circ(uv, float2(0.965445, 0.505026), 0.006348);
  ret += circ(uv, float2(0.514847, 0.865444), 0.00623523);
  ret += circ(uv, float2(0.710575, 0.0415131), 0.00322689);
  ret += circ(uv, float2(0.71403, 0.576945), 0.0215641);
  ret += circ(uv, float2(0.748873, 0.413325), 0.0110795);
  ret += circ(uv, float2(0.0623365, 0.896713), 0.0236203);
  ret += circ(uv, float2(0.980482, 0.473849), 0.00573439);
  ret += circ(uv, float2(0.647463, 0.654349), 0.0188713);
  ret += circ(uv, float2(0.651406, 0.981297), 0.00710875);
  ret += circ(uv, float2(0.428928, 0.382426), 0.0298806);
  ret += circ(uv, float2(0.811545, 0.62568), 0.00265539);
  ret += circ(uv, float2(0.400787, 0.74162), 0.00486609);
  ret += circ(uv, float2(0.331283, 0.418536), 0.00598028);
  ret += circ(uv, float2(0.894762, 0.0657997), 0.00760375);
  ret += circ(uv, float2(0.525104, 0.572233), 0.0141796);
  ret += circ(uv, float2(0.431526, 0.911372), 0.0213234);
  ret += circ(uv, float2(0.658212, 0.910553), 0.000741023);
  ret += circ(uv, float2(0.514523, 0.243263), 0.0270685);
  ret += circ(uv, float2(0.0249494, 0.252872), 0.00876653);
  ret += circ(uv, float2(0.502214, 0.47269), 0.0234534);
  ret += circ(uv, float2(0.693271, 0.431469), 0.0246533);
  ret += circ(uv, float2(0.415, 0.884418), 0.0271696);
  ret += circ(uv, float2(0.149073, 0.41204), 0.00497198);
  ret += circ(uv, float2(0.533816, 0.897634), 0.00650833);
  ret += circ(uv, float2(0.0409132, 0.83406), 0.0191398);
  ret += circ(uv, float2(0.638585, 0.646019), 0.0206129);
  ret += circ(uv, float2(0.660342, 0.966541), 0.0053511);
  ret += circ(uv, float2(0.513783, 0.142233), 0.00471653);
  ret += circ(uv, float2(0.124305, 0.644263), 0.00116724);
  ret += circ(uv, float2(0.99871, 0.583864), 0.0107329);
  ret += circ(uv, float2(0.894879, 0.233289), 0.00667092);
  ret += circ(uv, float2(0.246286, 0.682766), 0.00411623);
  ret += circ(uv, float2(0.0761895, 0.16327), 0.0145935);
  ret += circ(uv, float2(0.949386, 0.802936), 0.0100873);
  ret += circ(uv, float2(0.480122, 0.196554), 0.0110185);
  ret += circ(uv, float2(0.896854, 0.803707), 0.013969);
  ret += circ(uv, float2(0.292865, 0.762973), 0.00566413);
  ret += circ(uv, float2(0.0995585, 0.117457), 0.00869407);
  ret += circ(uv, float2(0.377713, 0.00335442), 0.0063147);
  ret += circ(uv, float2(0.506365, 0.531118), 0.0144016);
  ret += circ(uv, float2(0.408806, 0.894771), 0.0243923);
  ret += circ(uv, float2(0.143579, 0.85138), 0.00418529);
  ret += circ(uv, float2(0.0902811, 0.181775), 0.0108896);
  ret += circ(uv, float2(0.780695, 0.394644), 0.00475475);
  ret += circ(uv, float2(0.298036, 0.625531), 0.00325285);
  ret += circ(uv, float2(0.218423, 0.714537), 0.00157212);
  ret += circ(uv, float2(0.658836, 0.159556), 0.00225897);
  ret += circ(uv, float2(0.987324, 0.146545), 0.0288391);
  ret += circ(uv, float2(0.222646, 0.251694), 0.00092276);
  ret += circ(uv, float2(0.159826, 0.528063), 0.00605293);
	return std::max(ret, 0.0f);
}

// Procedural texture generation for the water
static inline float3 water(float2 uv, float3 cdir)
{
  const float iTime       = 0.5f;
  const float3 WATER_COL  = float3(0.0f, 0.4453f, 0.7305f);
  const float3 WATER2_COL = float3(0.0f, 0.4180f, 0.6758f);
  const float3 FOAM_COL   = float3(0.8125f, 0.9609f, 0.9648f);
  const float M_2PI = 6.283185307f;
  const float M_6PI = 18.84955592f;

  uv *= float2(0.25);

  // Parallax height distortion with two directional waves at
  // slightly different angles.
  float2 a = 0.025f * float2(cdir.x, cdir.z) / cdir.y; // Parallax offset
  float h = std::sin(uv.x + iTime); // Height at UV
  uv += a * h;
  h = std::sin(0.841471f * uv.x - 0.540302f * uv.y + iTime);
  uv += a * h;
    
  // Texture distortion
  float d1 = mod(uv.x + uv.y, M_2PI);
  float d2 = mod((uv.x + uv.y + 0.25f) * 1.3f, M_6PI);
  d1 = iTime * 0.07f + d1;
  d2 = iTime * 0.5f + d2;
  float2 dist = float2(
  	std::sin(d1) * 0.15f + std::sin(d2) * 0.05f,
  	std::cos(d1) * 0.15f + std::cos(d2) * 0.05f
  );
    
  float3 ret = mix(WATER_COL, WATER2_COL, waterlayer(uv + float2(dist.x, dist.y)));
  ret = mix(ret, FOAM_COL, waterlayer(float2(1.0f) - uv - float2(dist.y, dist.x)));
  return ret;
}

struct Ocean2D : public IProcTexture2D
{
  Ocean2D() { m_tag = GetTag(); }
  uint32_t GetTag()          const override { return TAG_OCEAN; }      
  float3 Evaluate(float2 tc) const override { return water((tc - float2(0.5f,0.5f))*16.0f, float3(0.01f,1.0f,0.01f)); }

  uint32_t m_dummy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Based off of iq's described here: http://www.iquilezles.org/www/articles/voronoilin
//

static inline float2 fract2(float2 v)
{
  return v - floor(v);
}

static inline float2 hash2(float2 p ) 
{
  const float2 p1 = float2(dot(p, make_float2(123.4f, 748.6f)), dot(p, float2(547.3f, 659.3f)));
  return fract2(float2(sin(p1.x)*5232.85324f, sin(p1.y)*5232.85324f));   
}

static inline float4 mix(float4 x, float4 y, float a)
{
  return x*(1.0f - a) + y*a;
}

static inline float voronoi(float2 p, float iTime) 
{
    float2 n = floor(p);
    float2 f = fract2(p);
    float md = 5.0f;
    float2 m = float2(0.0f, 0.0f);
    for (int i = -1;i<=1;i++) {
        for (int j = -1;j<=1;j++) {
            float2 g = float2(i, j);
            float2 o = hash2(n+g);
            o = 0.5f+0.5f*float2(std::sin(iTime + 5.038f*o.x), std::sin(iTime + 5.038f*o.y));
            float2 r = g + o - f;
            float d = dot(r, r);
            if (d<md) {
              md = d;
              m = n+g+o;
            }
        }
    }
    return md;
}

static inline float ov(float2 p, float iTime) 
{
    float v = 0.0f;
    float a = 0.4f;
    for (int i = 0;i<3;i++) {
        v+= voronoi(p, iTime)*a;
        p*=2.0f;
        a*=0.5f;
    }
    return v;
}

struct Voronoi2D : public IProcTexture2D
{
  Voronoi2D() { m_tag = GetTag(); }
  uint32_t GetTag() const override { return TAG_VORONOI; }      
  
  float3 Evaluate(float2 tc) const override 
  { 
    const float f = ov(tc*16.0f, 1.0f);
    return clamp(2*float3(f*f,f*f*f,f), 0.0f, 1.0f);
  }

  uint32_t m_dummy;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

static inline float3 fract3(float3 v)
{
  return v - floor3(v);
}

static inline float4 fract4(float4 v)
{
  return v - floor4(v);
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

//static inline float mix(float x, float y, float a)
//{
//  return x*(1.0f - a) + y*a;
//}

static inline float2 mix2(float2 x, float2 y, float a)
{
  return make_float2(mix(x.x, y.x, a), mix(x.y, y.y, a));
}

static inline float3 mix3(float3 x, float3 y, float a)
{
  return make_float3(mix(x.x, y.x, a), mix(x.y, y.y, a), mix(x.z, y.z, a));
}

static inline float4 mix4(float4 x, float4 y, float a)
{
  return make_float4(mix(x.x, y.x, a), mix(x.y, y.y, a), mix(x.z, y.z, a), mix(x.w, y.w, a));
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

static inline float rand(float n) { return fract(std::sin(n) * 43758.5453123f);}

// Classic Perlin noise
static inline float cnoise(float3 P)
{
  float3 Pi0 = floor3(P); // Integer part for indexing
  float3 Pi1 = Pi0 + make_float3(1.0, 1.0, 1.0); // Integer part + 1
  Pi0 = mod289f3(Pi0);
  Pi1 = mod289f3(Pi1);
  float3 Pf0 = fract3(P); // Fractional part for interpolation
  float3 Pf1 = Pf0 - make_float3(1.0, 1.0, 1.0); // Fractional part - 1.0
  float4 ix = make_float4(Pi0.x, Pi1.x, Pi0.x, Pi1.x);
  float4 iy = make_float4(Pi0.y, Pi0.y, Pi1.y, Pi1.y);
  float4 iz0 = make_float4(Pi0.z, Pi0.z, Pi0.z, Pi0.z);
  float4 iz1 = make_float4(Pi1.z, Pi1.z, Pi1.z, Pi1.z);

  float4 ixy = permute(permute(ix) + iy);
  float4 ixy0 = permute(ixy + iz0);
  float4 ixy1 = permute(ixy + iz1);

  float4 gx0 = ixy0 * (1.0 / 7.0);
  float4 gy0 = fract4(floor4(gx0) * (1.0 / 7.0)) - 0.5;
  gx0 = fract4(gx0);
  float4 gz0 = make_float4(0.5, 0.5, 0.5, 0.5) - abs4(gx0) - abs4(gy0);
  float4 sz0 = step4_(gz0, make_float4(0.0, 0.0, 0.0, 0.0));
  gx0 -= sz0 * (step4(0.0, gx0) - 0.5);
  gy0 -= sz0 * (step4(0.0, gy0) - 0.5);

  float4 gx1 = ixy1 * (1.0 / 7.0);
  float4 gy1 = fract4(floor4(gx1) * (1.0 / 7.0)) - 0.5;
  gx1 = fract4(gx1);
  float4 gz1 = make_float4(0.5, 0.5, 0.5, 0.5) - abs4(gx1) - abs4(gy1);
  float4 sz1 = step4_(gz1, make_float4(0.0, 0.0, 0.0, 0.0));
  gx1 -= sz1 * (step4_(float4(0.0f), gx1) - 0.5f);
  gy1 -= sz1 * (step4_(float4(0.0f), gy1) - 0.5f);

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
  float4 n_z = mix4(make_float4(n000, n100, n010, n110), make_float4(n001, n101, n011, n111), fade_xyz.z);
  float2 n_yz = mix2(make_float2(n_z.x, n_z.y), make_float2(n_z.z, n_z.w), fade_xyz.y);
  float n_xyz = mix(n_yz.x, n_yz.y, fade_xyz.x);
  return 2.2 * n_xyz;
}

static inline float octave(float3 pos, int octaves, float persistence)
{
  float total = 0.0f;
  float frequency = 10.0f;
  float amplitude = 3.0f;
  float maxValue = 0.0f;

  for (int i = 0; i < octaves; i++)
  {
    total += cnoise(pos * frequency) * amplitude;

    maxValue += amplitude;

    amplitude *= persistence;
    frequency *= 2;
  }

  return total / maxValue;
}

static inline float3 ramp(float3 color1, float3 color2, float3 colorramp, float pos,float pos_r )
{
	if (pos <= pos_r)
	{
		float r = mix(color1.x, colorramp.x, pos/pos_r);
		float g = mix(color1.y, colorramp.y, pos/pos_r);
		float b = mix(color1.z, colorramp.z, pos/pos_r);
		return make_float3(r, g, b);
	}
	else
	{
		float r = mix(colorramp.x, color2.x, (pos-pos_r)/(1.0f - pos_r));
		float g = mix(colorramp.y, color2.y, (pos-pos_r)/(1.0f - pos_r));
		float b = mix(colorramp.z, color2.z, (pos-pos_r)/(1.0f - pos_r));
		return make_float3(r, g, b);
	}
}

struct Perlin2D : public IProcTexture2D
{
  Perlin2D() { m_tag = GetTag(); }
  uint32_t GetTag() const override { return TAG_PERLIN; }      
  
  float3 Evaluate(float2 tc) const override 
  { 
    float n1 = octave(1.0f * float3(tc.x, tc.y, tc.x + tc.y), 4, 0.7f);
    float n2 = octave(1.0f * float3(tc.y, tc.x, (tc.x + tc.y) / std::max(abs(tc.x - tc.y),0.1f)), 6, 2.5f);
    float n3 = octave(2.0f * float3(tc.x+tc.y, tc.y-tc.x, tc.x*tc.y), 6, 1.5f);
    return clamp(float3(n1, n2, n3)*5.0f, 0.0f, 1.0f);
  }

  uint32_t m_dummy;
};

