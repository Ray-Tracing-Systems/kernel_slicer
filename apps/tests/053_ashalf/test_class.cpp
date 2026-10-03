#include "test_class.h"

using LiteMath::float2;
using LiteMath::as_half;
using LiteMath::as_uint16;
using LiteMath::as_uint;
using LiteMath::as_int;
using LiteMath::as_float;
using LiteMath::bit_cast;
using LiteMath::f32tof16;
using LiteMath::f16tof32;

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static inline float PackHalf2(float2 v)
{
  uint32_t lo = f32tof16(v.x);
  uint32_t hi = f32tof16(v.y);
  return as_float((hi << 16) | lo);
}

static inline float2 UnpackHalf2(float packedVal)
{
  uint32_t bits = as_uint(packedVal);
  return float2(f16tof32(bits & 0xFFFF), f16tof32((bits >> 16) & 0xFFFF));
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Test2D::Test2D(size_t a_size)
{
  m_testPixels.resize(a_size);
  m_testPixels2.resize(a_size);
  m_testPixels3.resize(a_size);
  for(uint32_t i=0;i<m_testPixels.size();i++) {
    LBPixel px;
    px.color[0] = half(1.0f*float(i));
    px.color[1] = half(0.5f*float(i));
    px.color[2] = half(0.25f*float(i));
    px.index    = uint16_t(i);
    m_testPixels [i] = px;
    m_testPixels2[i] = half4(px.color[0], px.color[1], px.color[2], as_half(px.index));
    m_testPixels3[i] = float2(PackHalf2(float2(0.5f*float(i), 0.25f*float(i))), as_float(i));
  }
}

void Test2D::kernel1D_Eval(const int a_size, float4* outData4f)
{
  for(int i=0;i<a_size;i++)
  {
    LBPixel test = m_testPixels[i];
    half4   test2= m_testPixels2[i];
    float2  test3= m_testPixels3[i];

    float4  val  = float4(test.color[0], test.color[1], test.color[2], float(test.index));
    float4  val2 = -float4(test2.x, test2.y, test2.z, float(bit_cast<uint16_t>(test2.w)));
    half testval3 = as_half(uint16_t(32));
    
    float2 yz  = UnpackHalf2(test3.x);

    if(i < a_size/3)    
      outData4f[i] = val*2.0f;
    else if(i < 2*a_size/3)
      outData4f[i] = val2*2.0f;
    else 
      outData4f[i] = float4(-1.f, yz.x, yz.y, as_uint(test3.y))*2.0f;
  }
}

void Test2D::Run(int a_size, float4* outData4f)
{
  kernel1D_Eval(a_size,outData4f);
}