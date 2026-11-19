#include "test_class.h"

void ImageArgs::SetSize(int a_width, int a_height)
{
  m_input.resize(a_width, a_height);
  m_temp.resize(a_width, a_height);
  for(int y = 0; y < a_height; y++)
    for(int x = 0; x < a_width; x++)
      m_input[int2(x, y)] = float4(float(3*x + y));
}

void ImageArgs::Run(int a_width, int a_height, const Image2D<float4>& a_input, float* a_output)
{
  kernel2D_Combine(a_width, a_height, a_input, m_input, m_temp);
  kernel2D_Read(a_width, a_height, m_temp, a_output);
}

void ImageArgs::kernel2D_Combine(int a_width, int a_height, const Image2D<float4>& a_first, const Image2D<float4>& a_second, Image2D<float4>& a_target)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      const int2 pos(x, y);
      a_target[pos] = a_first[pos]*2.0f + a_second[pos];
    }
  }
}

void ImageArgs::kernel2D_Read(int a_width, int a_height, const Image2D<float4>& a_source, float* a_output)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      a_output[y*a_width + x] = a_source[int2(x, y)].x;
    }
  }
}
