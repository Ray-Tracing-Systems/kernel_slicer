#include "test_class.h"
#include "Bitmap.h"
#include "texture2d.h"
#include "sampler.h"
#include <cassert>

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
  std::vector<uint32_t> green_white = {GREEN, WHITE, WHITE, GREEN };

  std::vector<uint32_t> blue_cross  = {BLACK, BLACK, BLUE, BLUE, BLACK, BLACK,
                                       BLACK, BLACK, BLUE, BLUE, BLACK, BLACK,
                                       BLUE, BLUE, BLUE, BLUE, BLUE, BLUE,
                                       BLUE, BLUE, BLUE, BLUE, BLUE, BLUE,
                                       BLACK, BLACK, BLUE, BLUE, BLACK, BLACK,
                                       BLACK, BLACK, BLUE, BLUE, BLACK, BLACK};

  std::vector<uint32_t> yellow_diamond = {BLACK, YELLOW, BLACK,
                                          YELLOW, BLACK, YELLOW,
                                          BLACK, YELLOW, BLACK };

  std::shared_ptr< Texture2D<uint32_t> > pTexture1 = std::make_shared< Texture2D<uint32_t> >(3, 3, red_black_white.data());
  std::shared_ptr< Texture2D<uint32_t> > pTexture2 = std::make_shared< Texture2D<uint32_t> >(2, 2, green_white.data());
  std::shared_ptr< Texture2D<uint32_t> > pTexture3 = std::make_shared< Texture2D<uint32_t> >(6, 6, blue_cross.data());
  std::shared_ptr< Texture2D<uint32_t> > pTexture4 = std::make_shared< Texture2D<uint32_t> >(3, 3, yellow_diamond.data());

  Sampler sampler;
  sampler.filter   = Sampler::Filter::NEAREST; 
  sampler.addressU = Sampler::AddressMode::CLAMP;
  sampler.addressV = Sampler::AddressMode::CLAMP;

  m_textures.push_back(MakeCombinedTexture2D(pTexture1, sampler));
  m_textures.push_back(MakeCombinedTexture2D(pTexture2, sampler));
  m_textures.push_back(MakeCombinedTexture2D(pTexture3, sampler));
  m_textures.push_back(MakeCombinedTexture2D(pTexture4, sampler));

  m_textures2.resize(4);
  m_textures2[0] = m_textures[1].get();
  m_textures2[1] = m_textures[2].get();
  m_textures2[2] = m_textures[3].get();
  m_textures2[3] = m_textures[0].get();
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
      const float2 uv = get_uv(x, y, a_width, a_height);
      float2 c = float2(-.445, 0.0) + (uv - float2(0.5)) * (2.0f + 1.7f * 0.2f);
      float2 z = float2(0.0);
      int    n = 0;
      for (int i = 0; i < FRACTAL_ITERATIONS; i++)
      {
        z = float2(z.x * z.x - z.y * z.y, 2.0f * z.x * z.y) + c;
        if (dot(z, z) > 2.0f) 
          break;
        n++;
      }
      
      float4 color;
      if(uv.x < 0.75f)
        color = m_textures[n%4]->sample(uv); 
      else
        color = m_textures2[n%4]->sample(uv);
      outData1ui[y*a_width + x] = RealColorToUint32(color, 2.2f);
    }
  }
}