#include "mandelbrot.h"
#include <algorithm>
#include <chrono>

#ifndef unmasked
#define unmasked 
#endif

static inline int mandel(float c_re, float c_im, int count) 
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
      if (z_re * z_re + z_im * z_im > 4.)
          break;
      float new_re = z_re*z_re - z_im*z_im;
      float new_im = 2.f * z_re * z_im;
      unmasked {
          z_re = c_re + new_re;
          z_im = c_im + new_im;
      }
  }
  return i;
}

void Mandelbrot::kernel2D_process(int w, int h, uint32_t* outData)
{
  #pragma omp parallel for 
  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++) 
    {  
      const float tx  = 0.5f*((float)x - (0.75f * (float)w)) / (w / 4);
      const float ty  = 0.5f*((float)y - (h / 2)) / (h / 4);
      const int index = mandel(tx, ty, 100);

      int r1 = min((index*128)/32, 255);
      int g1 = min((index*128)/25, 255);
      int b1 = min((index*index), 255);

      outData[y*w+x] = 0xFF000000 | (r1) | (g1 << 8) | (b1 << 16);
    }
  }
}

void Mandelbrot::Fractal(int w, int h, uint32_t* outData)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel2D_process(w, h, outData);  
  m_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}