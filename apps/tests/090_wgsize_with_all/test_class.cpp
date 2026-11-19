#include "test_class.h"

void WgSizeAll::Run(const float* a_in, float* a_out, uint32_t a_size)
{
  kernel1D_Scale(a_in, a_out, a_size);
  kernel2D_Fill(32, 4, a_out);
}

void WgSizeAll::kernel1D_Scale(const float* a_in, float* a_out, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
    a_out[i] = a_in[i] * 2.0f - 1.0f;
}

void WgSizeAll::kernel2D_Fill(uint32_t a_width, uint32_t a_height, float* a_out)
{
  for(uint32_t y = 0; y < a_height; y++)
    for(uint32_t x = 0; x < a_width; x++)
      a_out[y*a_width + x] += float(x + y);
}
