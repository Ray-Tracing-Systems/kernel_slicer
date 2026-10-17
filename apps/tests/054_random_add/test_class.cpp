#include "test_class.h"

#include "LiteMath.h"
using LiteMath::InterlockedAdd;
using LiteMath::float3;
using LiteMath::InterlockedAdd3f;

void RandomAdd::AddOnes(uint32_t a_iterNum, float* a_out, uint32_t a_size, uint32_t a_sameNum)
{
  kernel1D_AddOnes(a_iterNum, a_out, a_size, a_sameNum);
}

void RandomAdd::kernel1D_AddOnes(uint32_t a_iterNum, float* a_out, uint32_t a_size, uint32_t a_sameNum)
{
  #pragma omp parallel for
  for(uint32_t i=0; i < a_iterNum; i++)
  {
    const uint32_t addr = PseudoRandomIndex(i / a_sameNum, a_size);
    const float3   val  = float3(1.0f, 2.0f, 3.0f);
    InterlockedAdd3f(a_out, int(addr*4), val);
  }
}
