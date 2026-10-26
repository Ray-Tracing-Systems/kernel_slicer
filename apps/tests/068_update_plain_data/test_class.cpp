#include "test_class.h"

void PlainUpdate::Run(const float* a_in, float* a_out, uint32_t a_size)
{
  kernel1D_Transform(a_in, a_out, a_size);
}

void PlainUpdate::kernel1D_Transform(const float* a_in, float* a_out, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
  {
    const uint32_t tableId = (i + m_shift) % uint32_t(m_table.size());
    a_out[i] = a_in[i]*m_scale + m_offset + m_table[tableId];
  }
}
