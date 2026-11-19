#include "test_class.h"

void IfElseCalls::Run(const float* a_in, float* a_out, uint32_t a_size)
{
  if(m_mode == 0)
    kernel1D_Add(a_in, a_out, a_size);
  else if(m_mode == 1)
    kernel1D_Mul(a_in, a_out, a_size);
  else
    kernel1D_Add(a_in, a_out, a_size);

  if(m_mode == 2)
    kernel1D_Neg(a_out, a_size); // 'if' without 'else' and without braces: the kernel must not run in other modes
}

void IfElseCalls::kernel1D_Add(const float* a_in, float* a_out, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
    a_out[i] = a_in[i] + 1.0f;
}

void IfElseCalls::kernel1D_Mul(const float* a_in, float* a_out, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
    a_out[i] = a_in[i] * 3.0f;
}

void IfElseCalls::kernel1D_Neg(float* a_data, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
    a_data[i] = -a_data[i];
}
