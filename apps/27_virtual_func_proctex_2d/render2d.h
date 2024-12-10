#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>

#include "LiteMath.h"
using namespace LiteMath;

struct IProcTexture2D
{
  static constexpr uint32_t TAG_EMPTY      = 0;   
  static constexpr uint32_t TAG_COLOR_RED  = 1;
  static constexpr uint32_t TAG_MANDELBROT = 2;
  static constexpr uint32_t TAG_OCEAN      = 3; 

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

  static constexpr uint32_t TOTAL_IMPLEMANTATIONS = 3;

  enum {BRANCHING_LITE = 0, BRANCHING_MEDIUM = 1, BRANCHING_HEAVY = 2 }; 

protected:

  virtual void kernel2D_EvaluateTextures(int w, int h, uint32_t* outData, int a_branchMode);
  float m_time;

  std::vector<IProcTexture2D*> allProcTextures; 
  void InitAllTextures();
};


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


struct Red2D : public IProcTexture2D
{
  Red2D() { m_tag = GetTag(); }  

  uint32_t GetTag()          const override { return TAG_COLOR_RED; }      
  float3 Evaluate(float2 tc) const override { return float3(1.0f, 0.0f, 0.0f); }
};

struct Mandelbrot2D : public IProcTexture2D
{
  Mandelbrot2D() { m_tag = GetTag(); }
  uint32_t GetTag() const override { return TAG_MANDELBROT; }      
  
  float3 Evaluate(float2 tc) const override 
  { 
    const int index = mandel(tc.x-0.5f, tc.y, 100);

    const int r1 = std::min((index*128)/32, 255);
    const int g1 = std::min((index*128)/25, 255);
    const int b1 = std::min((index*index), 255);

    const float fr1 = float(r1)/255.0f;
    const float fg1 = float(g1)/255.0f;
    const float fb1 = float(b1)/255.0f;

    return float3(fr1, fg1, fb1); 
  }
};

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
  uint32_t GetTag() const override { return TAG_OCEAN; }      
  
  float3 Evaluate(float2 tc) const override 
  { 
    return water((tc - float2(0.5f,0.5f))*16.0f, float3(0.01f,1.0f,0.01f));
  }
};
