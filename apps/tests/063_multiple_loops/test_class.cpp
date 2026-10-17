#include "test_class.h"

void MultiLoops::kernel1D_TwoPasses(const float* a_in, float* a_out, uint32_t a_size)
{
  const float scale = 0.5f;                  // single-threaded section #0
  for(uint32_t i=0; i<a_size; i++)           // parallel loop #0
    m_tmp[i] = a_in[i]*scale;

  float bias = m_tmp[1] + 1.0f;              // single-threaded section #1, reads results of loop #0
  m_count = a_size/2;
  m_summ  = 0.0f;
  for(uint32_t i=0; i<m_count; i++)          // parallel loop #1, size is known only on GPU (indirect dispatch), reduction
  {
    const float val = m_tmp[2*i] + m_tmp[2*i+1] + bias;
    a_out[i] = val;
    m_summ  += val;
  }

  m_result = bias*2.0f + scale;              // single-threaded epilogue
}

void MultiLoops::kernel2D_Blur(const float* a_in, float* a_out, uint32_t a_width, uint32_t a_height)
{
  for(uint32_t y=0; y<a_height; y++)         // horizontal pass
  {
    for(uint32_t x=0; x<a_width; x++)
    {
      const uint32_t x0 = (x == 0) ? 0 : x-1;
      const uint32_t x1 = std::min(x+1, a_width-1);
      m_tmp[y*a_width + x] = (a_in[y*a_width + x0] + a_in[y*a_width + x] + a_in[y*a_width + x1])*(1.0f/3.0f);
    }
  }

  for(uint32_t y=0; y<a_height; y++)         // vertical pass
  {
    for(uint32_t x=0; x<a_width; x++)
    {
      const uint32_t y0 = (y == 0) ? 0 : y-1;
      const uint32_t y1 = std::min(y+1, a_height-1);
      a_out[y*a_width + x] = (m_tmp[y0*a_width + x] + m_tmp[y*a_width + x] + m_tmp[y1*a_width + x])*(1.0f/3.0f);
    }
  }
}

void MultiLoops::Run(const float* a_in, float* a_out, uint32_t a_size)
{
  kernel1D_TwoPasses(a_in, a_out, a_size);
}

void MultiLoops::Blur(const float* a_in, float* a_out, uint32_t a_width, uint32_t a_height)
{
  kernel2D_Blur(a_in, a_out, a_width, a_height);
}
