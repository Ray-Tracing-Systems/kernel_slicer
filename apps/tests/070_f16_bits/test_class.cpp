#include "test_class.h"

#include "LiteMath.h"
using LiteMath::f32tof16;
using LiteMath::f16tof32;

void SimpleTest::Convert(uint32_t a_size, const float* a_in, uint32_t* a_bits, float* a_out)
{
  kernel1D_Convert(a_size, a_in, a_bits, a_out);
}

void SimpleTest::kernel1D_Convert(uint32_t a_size, const float* a_in, uint32_t* a_bits, float* a_out) // 'f32tof16/f16tof32' must not need 16 bit types in shaders
{
  for(uint32_t i=0; i < a_size; i++)
  {
    const uint32_t bits = f32tof16(a_in[i]);
    a_bits[i] = bits;
    a_out[i]  = f16tof32(bits);
  }
}
