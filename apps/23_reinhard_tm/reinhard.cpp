#include "reinhard.h"
#include <algorithm>

float reinhard_extended(float v, float max_white)
{
  float numerator = v * (1.0f + (v /(max_white * max_white)));
  return numerator / (1.0f + v);
}

void ReinhardTM::kernel1D_findMax(int size, const float* inData4f)
{
  m_whitePoint = 0;  
  for(int i=0;i<size;i++)
  {
    float r = inData4f[4*i+0];
    float g = inData4f[4*i+1];
    float b = inData4f[4*i+2];

    float maxVal = std::max(r, std::max(g,b));
    m_whitePoint = std::max(m_whitePoint, maxVal);
  }

}

void ReinhardTM::kernel2D_process(int w, int h, const float* inData4f, uint32_t* outData)
{
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++) 
    { 
      int   i = y*w+x;
      float r = reinhard_extended(inData4f[4*i+0], m_whitePoint);
      float g = reinhard_extended(inData4f[4*i+1], m_whitePoint);
      float b = reinhard_extended(inData4f[4*i+2], m_whitePoint);

      uint32_t r1 = (uint32_t)( std::min(r*255.0f, 255.0f)  );
      uint32_t g1 = (uint32_t)( std::min(g*255.0f, 255.0f)  );
      uint32_t b1 = (uint32_t)( std::min(b*255.0f, 255.0f)  );

      outData[y*w+x] = 0xFF000000 | (r1) | (g1 << 8) | (b1 << 16);
    }
  }
}

void ReinhardTM::Run(int w, int h, const float* inData4f, uint32_t* outData)
{
  kernel1D_findMax(w*h, inData4f);
  kernel2D_process(w,h, inData4f, outData);  
}