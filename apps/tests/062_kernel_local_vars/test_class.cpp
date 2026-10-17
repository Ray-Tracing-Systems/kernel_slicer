#include "test_class.h"

void KernelLocals::kernel1D_Process(const float* a_in, float* a_out, uint32_t a_size, float a_gamma)
{
  const float invGamma = 1.0f/a_gamma;
  float maxW   = std::max(m_weights[0], m_weights[1]);
  float offset = m_scale*0.5f;
  offset += maxW;
  for(uint32_t i=0; i<a_size; i++)
    a_out[i] = std::pow(a_in[i], invGamma)*maxW + offset;
  m_result = offset + invGamma;
}

void KernelLocals::kernel2D_Gradient(float* a_out, uint32_t a_width, uint32_t a_height)
{
  float2 invSize = float2(1.0f/float(a_width), 1.0f/float(a_height));
  float  bias    = m_weights[2];
  for(uint32_t y=0; y<a_height; y++)
  {
    for(uint32_t x=0; x<a_width; x++)
      a_out[y*a_width + x] = float(x)*invSize.x + float(y)*invSize.y + bias;
  }
}

void KernelLocals::Run(const float* a_in, float* a_out, uint32_t a_size, float a_gamma)
{
  kernel1D_Process(a_in, a_out, a_size, a_gamma);
}

void KernelLocals::Run2D(float* a_out, uint32_t a_width, uint32_t a_height)
{
  kernel2D_Gradient(a_out, a_width, a_height);
}
