#include "render2d.h"
#include <algorithm>
#include <chrono>

#ifndef unmasked
#define unmasked 
#endif

static inline uint RealColorToUint32_f3(float3 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  uint red = (uint)r, green = (uint)g, blue = (uint)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

ProcRender2D::ProcRender2D()
{
  InitAllTextures();
}

ProcRender2D::~ProcRender2D()
{
  for(auto& tex : allProcTextures)
    delete tex;
  allProcTextures.clear();
}

void ProcRender2D::InitAllTextures()
{
  allProcTextures.push_back(new Red2D);
  allProcTextures.push_back(new Mandelbrot2D);
}

void ProcRender2D::kernel2D_EvaluateTextures(int w, int h, uint32_t* outData)
{
  #pragma omp parallel for 
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++) 
    { 
      const float2 texCoord = float2(float(x)/float(w), float(y)/float(h));
      const int index       = (x + y) % TOTAL_IMPLEMANTATIONS;
      const float3 color    = allProcTextures[index]->Evaluate(texCoord);

      outData[y*w+x] = RealColorToUint32_f3(color);

      //const float tx  = 0.5f*((float)x - (0.75f * (float)w)) / (w / 4);
      //const float ty  = 0.5f*((float)y - (h / 2)) / (h / 4);
      //const int index = mandel(tx, ty, 100);
      //
      //int r1 = min((index*128)/32, 255);
      //int g1 = min((index*128)/25, 255);
      //int b1 = min((index*index), 255);
      //
      //outData[y*w+x] = 0xFF000000 | (r1) | (g1 << 8) | (b1 << 16);
    }
  }
}

void ProcRender2D::Fractal(int w, int h, uint32_t* outData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_EvaluateTextures(w, h, outData);  
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}