#include "reinhard.h"
#include <algorithm>
#include <chrono>

float reinhard_extended(float v, float max_white)
{
  float numerator = v * (1.0f + (v / (max_white * max_white)));
  return numerator / (1.0f + v);
}

void ReinhardTM::kernel1D_finMax(const float* hdrData, int size)
{
  whitePoint = 0.0;
  for(int i=0;i<size;i++)
  {
    float r = hdrData[4*i+0];
    float g = hdrData[4*i+1];
    float b = hdrData[4*i+2];
    float maxValue = std::max(r, std::max(g,b));
    whitePoint = std::max(whitePoint, maxValue);
  }  

}

void ReinhardTM::kernel2D_process(int w, int h, const float* hdrData, uint32_t* ldrData)
{
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++)
    {
      float r = reinhard_extended(hdrData[4*(y*w+x)+0], whitePoint);
      float g = reinhard_extended(hdrData[4*(y*w+x)+1], whitePoint);
      float b = reinhard_extended(hdrData[4*(y*w+x)+2], whitePoint);
      
      int ir  = (int)std::min(r*255.0f, 255.0f);
      int ig  = (int)std::min(g*255.0f, 255.0f);
      int ib  = (int)std::min(b*255.0f, 255.0f);
      
      ldrData[y*w+x] = 0xFF000000 | (ib << 16) | (ig << 8) | ir;
    }
  }
}

void ReinhardTM::Run(int w, int h, const float* hdrData, uint32_t* ldrData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel1D_finMax(hdrData, w*h);
  kernel2D_process(w,h,hdrData,ldrData);
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

