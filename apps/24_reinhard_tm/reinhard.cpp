#include "reinhard.h"
#include <algorithm>
#include <chrono>

float reinhard_extended(float v, float max_white)
{
  float numerator = v * (1.0f + (v /(max_white * max_white)));
  return numerator / (1.0f + v);
}

void ReinhardTM::kernel1D_findMax(int size, const float4* inData4f)
{
  m_whitePoint = 0;  
  #pragma omp parallel for reduction(max:m_whitePoint)
  for(int i=0;i<size;i++)
  {
    float4 color = inData4f[i];
    float r = color.x;
    float g = color.y;
    float b = color.z;
    float maxVal = std::max(r, std::max(g,b));
    m_whitePoint = std::max(m_whitePoint, maxVal);
  }

}

void ReinhardTM::kernel2D_process(int w, int h, const float4* inData4f, uint32_t* outData)
{
  #pragma omp parallel for 
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++) 
    { 
      float4 color = inData4f[y*w+x];
      float r = reinhard_extended(color.x, m_whitePoint);
      float g = reinhard_extended(color.y, m_whitePoint);
      float b = reinhard_extended(color.z, m_whitePoint);

      int r1 = (int)( std::min(r*255.0f, 255.0f)  );
      int g1 = (int)( std::min(g*255.0f, 255.0f)  );
      int b1 = (int)( std::min(b*255.0f, 255.0f)  );

      outData[y*w+x] = 0xFF000000 | (r1) | (g1 << 8) | (b1 << 16);
    }
  }
}

void ReinhardTM::Run(int w, int h, const float4* inData4f, uint32_t* outData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel1D_findMax(w*h, inData4f);
  kernel2D_process(w,h, inData4f, outData);  
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}