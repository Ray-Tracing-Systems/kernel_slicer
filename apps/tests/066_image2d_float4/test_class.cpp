#include "test_class.h"

static inline uint32_t ToUint32(float4 a_color)
{
  const uint32_t r = uint32_t(clamp(a_color.x, 0.0f, 1.0f) * 255.0f);
  const uint32_t g = uint32_t(clamp(a_color.y, 0.0f, 1.0f) * 255.0f);
  const uint32_t b = uint32_t(clamp(a_color.z, 0.0f, 1.0f) * 255.0f);
  return (r << 16) | (g << 8) | b;
}

void TexFilter::SetSize(int a_width, int a_height)
{
  m_smallWidth  = a_width/4;
  m_smallHeight = a_height/4;
  m_bright.resize(a_width, a_height);
  m_small.resize(m_smallWidth, m_smallHeight);
  m_sampler.filter   = Sampler::Filter::LINEAR;
  m_sampler.addressU = Sampler::AddressMode::CLAMP;
  m_sampler.addressV = Sampler::AddressMode::CLAMP;
}

void TexFilter::Run(int a_width, int a_height, const Image2D<float4>& a_input, uint32_t* a_out)
{
  kernel2D_ExtractBright(a_width, a_height, a_input, m_bright);
  kernel2D_DownSample4x(m_smallWidth, m_smallHeight, m_bright, m_small);
  kernel2D_Mix(a_width, a_height, a_input, a_out);
}

void TexFilter::kernel2D_ExtractBright(int a_width, int a_height, const Image2D<float4>& a_in, Image2D<float4>& a_out)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      float4 color = a_in[int2(x, y)];
      if(color.x + color.y + color.z < m_threshold*3.0f)
        color = float4(0.0f, 0.0f, 0.0f, 0.0f);
      a_out[int2(x, y)] = color;
    }
  }
}

void TexFilter::kernel2D_DownSample4x(int a_width, int a_height, const Image2D<float4>& a_in, Image2D<float4>& a_out)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      float4 average = float4(0.0f, 0.0f, 0.0f, 0.0f);
      for(int j = 0; j < 4; j++)
        for(int i = 0; i < 4; i++)
          average += a_in[int2(x*4 + i, y*4 + j)];
      a_out[int2(x, y)] = average*(1.0f/16.0f);
    }
  }
}

void TexFilter::kernel2D_Mix(int a_width, int a_height, const Image2D<float4>& a_in, uint32_t* a_out)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      const float2 uv    = float2((float(x) + 0.5f)/float(a_width), (float(y) + 0.5f)/float(a_height));
      const float4 bloom = m_small.sample(m_sampler, uv);
      a_out[y*a_width + x] = ToUint32(a_in[int2(x, y)]*0.5f + bloom);
    }
  }
}
