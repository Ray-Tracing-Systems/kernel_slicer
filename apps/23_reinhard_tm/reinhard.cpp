#include "reinhard.h"
#include <algorithm>

static inline float reinhard_extended(float v, float max_white)
{
  float numerator = v * (1.0f + (v / float(max_white * max_white)));
  return numerator / (1.0f + v);
}

void ReinhardTM::kernel1D_findMax(int size, const float* inData4f)
{
  m_whitePoint = 0.0f;
  for(int i=0;i<size;i++)
  {
    float maxColor = std::max(inData4f[i*4+0], std::max(inData4f[i*4+1], inData4f[i*4+2]));
    m_whitePoint = std::max(m_whitePoint, maxColor);
  }
}

void ReinhardTM::kernel2D_doToneMapping(int w, int h, const float* inData4f, unsigned int* outData)
{
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++)
    {
      int offset  = (y*w+x)*4;
      float red   = reinhard_extended(inData4f[offset+0], m_whitePoint);
      float green = reinhard_extended(inData4f[offset+1], m_whitePoint);
      float blue  = reinhard_extended(inData4f[offset+2], m_whitePoint);

      uint32_t r = uint32_t( std::min(red*255.0f,   255.0f) );
      uint32_t g = uint32_t( std::min(green*255.0f, 255.0f) );
      uint32_t b = uint32_t( std::min(blue*255.0f,  255.0f) );

      outData[y*w+x] = 0xFF000000 | r | (g << 8) | (b << 16);
    }
  }
}

void ReinhardTM::Run(int w, int h, const float* inData4f, unsigned int* outData)
{
  kernel1D_findMax(w*h, inData4f);
  kernel2D_doToneMapping(w,h,inData4f, outData);
}