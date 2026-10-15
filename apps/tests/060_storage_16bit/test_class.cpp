#include "test_class.h"

void Storage16::kernel1D_Unpack(float* a_out, uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++)
  {
    const PackedVertex vert = m_verts[i];
    const PackedVertex bias = m_bias;
    a_out[i] = float(uint32_t(bias.v + bias.flags[0])) + float(uint32_t(vert.u))*(1.0f/65535.0f) + float(uint32_t(vert.v)) + float(vert.id)*1000.0f + float(uint32_t(vert.flags[1]));
  }
}

void Storage16::Run(float* a_out, uint32_t a_size)
{
  kernel1D_Unpack(a_out, a_size);
}
