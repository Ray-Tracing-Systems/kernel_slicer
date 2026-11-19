#include "test_class.h"

void ImageArrayArgs::SetSize(int a_width, int a_height)
{
  for(auto& image : m_images)
    image.resize(a_width, a_height);
  for(int y = 0; y < a_height; y++)
    for(int x = 0; x < a_width; x++)
    {
      m_images[0][int2(x, y)] = float4(float(3*x + y + 1));
      m_images[2][int2(x, y)] = float4(float(x + 5*y + 2));
    }
}

void ImageArrayArgs::Run(int a_width, int a_height, float* a_output)
{
  constexpr int firstSource = 2;
  kernel2D_Scale(a_width, a_height, m_images[firstSource], m_images[1]);
  kernel2D_Scale(a_width, a_height, m_images[0], m_images[3]);
  kernel2D_Read(a_width, a_height, m_images[1], m_images[3], a_output);
}

void ImageArrayArgs::kernel2D_Scale(int a_width, int a_height, const Image2D<float4>& a_source, Image2D<float4>& a_target)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      const int2 pos(x, y);
      a_target[pos] = a_source[pos]*2.0f;
    }
  }
}

void ImageArrayArgs::kernel2D_Read(int a_width, int a_height, const Image2D<float4>& a_first, const Image2D<float4>& a_second, float* a_output)
{
  for(int y = 0; y < a_height; y++)
  {
    for(int x = 0; x < a_width; x++)
    {
      const int2 pos(x, y);
      a_output[y*a_width + x] = a_first[pos].x - a_second[pos].x;
    }
  }
}
