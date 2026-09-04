#include "test_class.h"

using LiteMath::as_half;
using LiteMath::as_uint16;
using LiteMath::as_uint;
using LiteMath::as_int;

Test2D::Test2D(size_t a_size)
{
  m_testPixels.resize(a_size);
  m_testPixels2.resize(a_size);
  for(size_t i=0;i<m_testPixels.size();i++) {
    LBPixel px;
    px.color[0] = half(1.0f*float(i));
    px.color[1] = half(0.5f*float(i));
    px.color[2] = half(0.25f*float(i));
    px.index    = uint16_t(i);
    m_testPixels [i] = px;
    m_testPixels2[i] = half4(px.color[0], px.color[1], px.color[2], as_half(px.index));
  }
}

void Test2D::kernel1D_Eval(const int a_size, float4* outData4f)
{
  for(int i=0;i<a_size;i++)
  {
    LBPixel test = m_testPixels[i];
    half4 test2  = m_testPixels2[i];

    float4  val  = float4(test.color[0], test.color[1], test.color[2], float(test.index));
    float4  val2 = -float4(test2.x, test2.y, test2.z, float(as_uint16(test2.w)));
    half test3 = as_half(uint16_t(32));

    if(i < a_size/2)    
      outData4f[i] = val*2.0f;
    else
      outData4f[i] = val2*2.0f;
  }
}

void Test2D::Run(int a_size, float4* outData4f)
{
  kernel1D_Eval(a_size,outData4f);
}