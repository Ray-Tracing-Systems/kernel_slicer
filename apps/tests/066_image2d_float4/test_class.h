#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"

using LiteMath::float4;
using LiteMath::float2;
using LiteMath::int2;
using LiteMath::clamp;
using LiteImage::Image2D;
using LiteImage::Sampler;

// Image2D<float4> textures: control function argument, kernel arguments and class members,
// texel read/write via operator[](int2) and filtered read via sample(sampler, uv)
class TexFilter
{
public:
  TexFilter(){}

  void SetSize(int a_width, int a_height);

  virtual void Run(int a_width, int a_height, const Image2D<float4>& a_input, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  virtual void kernel2D_ExtractBright(int a_width, int a_height, const Image2D<float4>& a_in, Image2D<float4>& a_out);
  virtual void kernel2D_DownSample4x (int a_width, int a_height, const Image2D<float4>& a_in, Image2D<float4>& a_out);
  virtual void kernel2D_Mix          (int a_width, int a_height, const Image2D<float4>& a_in, uint32_t* a_out);

  Image2D<float4> m_bright;
  Image2D<float4> m_small;
  Sampler         m_sampler;
  int             m_smallWidth  = 0;
  int             m_smallHeight = 0;
  float           m_threshold   = 0.5f;
};
