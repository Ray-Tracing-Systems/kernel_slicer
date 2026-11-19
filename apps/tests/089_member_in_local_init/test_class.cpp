#include "test_class.h"

void LocalInit::Run(uint32_t a_width, uint32_t a_height, const float* a_in, float* a_out, uint32_t a_size)
{
  kernel2D_Window(a_width, a_height, a_in, a_out);
}

void LocalInit::kernel2D_Window(uint32_t a_width, uint32_t a_height, const float* a_in, float* a_out)
{
  for(uint32_t y = 0; y < a_height; y++)
  {
    for(uint32_t x = 0; x < a_width; x++)
    {
      const uint32_t idx  = (y + m_startY) * m_pitch + (x + m_startX);   // members in parentheses
      const uint32_t idx2 = ((y + m_startY) * m_pitch) + ((x) + m_startX);
      const float    w    = float((x + m_startX) % 7) + float(m_startY);
      a_out[idx] = a_in[idx2] * 2.0f + w;
    }
  }
}
