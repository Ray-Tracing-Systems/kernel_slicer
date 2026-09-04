#include "test_class.h"

Test2D::Test2D(size_t a_size)
{
  m_testPixels.resize(a_size);
  for(size_t i=0;i<m_testPixels.size();i++) {
    LBPixel px;
    px.color[0] = half(1.0f*float(i));
    px.color[1] = half(0.5f*float(i));
    px.color[2] = half(0.25f*float(i));
    px.index    = uint16_t(i);
    m_testPixels[i] = px;
  }
}

void Test2D::kernel1D_MemSet1(const int a_size, float4* outData4f)
{
  for(int i=0;i<a_size;i++)
  {
    LBPixel test = m_testPixels[i];
    float4  val  = float4(test.color[0], test.color[1], test.color[2], float(test.index));
    outData4f[i] = val*2.0f;
  }
}

void Test2D::Run(int a_size, float4* outData4f)
{
  kernel1D_MemSet1(a_size,outData4f);
}