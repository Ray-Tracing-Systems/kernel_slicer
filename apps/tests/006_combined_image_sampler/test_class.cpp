#include "test_class.h"
#include "Image2d.h"

#include <cassert>

using LiteImage::Image2D;
using LiteImage::Sampler;
using LiteImage::ICombinedImageSampler;
using namespace LiteMath;

inline float2 get_uv(const int x, const int y, const uint width, const uint height)
{
  const float u = (float)(x) / (float)(width);
  const float v = (float)(y) / (float)(height);
  return float2(u, v);
}

inline uint RealColorToUint32(float4 a_realColor, const float a_gamma)
{
  float  r = pow(clamp(a_realColor.x, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  g = pow(clamp(a_realColor.y, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  b = pow(clamp(a_realColor.z, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  a =     clamp(a_realColor.w, 0.0F, 1.0F)           * 255.0f;

  uint red   = (uint)r;
  uint green = (uint)g;
  uint blue  = (uint)b;
  uint alpha = (uint)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

TestCombinedImage::TestCombinedImage()
{
  constexpr uint32_t WHITE  = 0x00FFFFFF;
  constexpr uint32_t BLACK  = 0x00000000;
  constexpr uint32_t RED    = 0x000000FF;
  constexpr uint32_t GREEN  = 0x0000FF00;
  constexpr uint32_t BLUE   = 0x00FF0000;
  constexpr uint32_t YELLOW = 0x0000FFFF;

  std::vector<uint32_t> red_black_white = {RED, BLACK, WHITE,
                                           BLACK, WHITE, RED,
                                           WHITE, RED, BLACK };

  std::vector<uint32_t> green_white = {GREEN, YELLOW, YELLOW, GREEN };                                         

  std::shared_ptr< Image2D<uint32_t> > pTexture  = std::make_shared< Image2D<uint32_t> >(3, 3, red_black_white.data());
  std::shared_ptr< Image2D<uint32_t> > pTexture2 = std::make_shared< Image2D<uint32_t> >(2, 2, green_white.data());

  Sampler sampler;
  sampler.filter   = Sampler::Filter::NEAREST; 
  sampler.addressU = Sampler::AddressMode::CLAMP;
  sampler.addressV = Sampler::AddressMode::CLAMP;

  m_pCombinedImage         = MakeCombinedTexture2D(pTexture, sampler);
  m_pCombinedImage2Storage = MakeCombinedTexture2D(pTexture2, sampler);
  m_pCombinedImage2        = m_pCombinedImage2Storage.get();
}

void TestCombinedImage::Run(const int a_width, const int a_height, unsigned int* outData1ui)
{
  kernel2D_Run(a_width, a_height, outData1ui);
}

void TestCombinedImage::kernel2D_Run(const int a_width, const int a_height, unsigned int* outData1ui)
{
  #pragma omp parallel for
  for (int y = 0; y < a_height; ++y)
  {
    for (int x = 0; x < a_width; ++x)
    {  
      const float2 uv    = get_uv(x, y, a_width, a_height);

      float4 color;

      if(uv.x < 0.75f)
        color = m_pCombinedImage->sample(uv*1.0f);
      else
        color = m_pCombinedImage2->sample(uv*1.0f);
      outData1ui[y*a_width + x] = RealColorToUint32(color, 2.2f);
    }
  }
}