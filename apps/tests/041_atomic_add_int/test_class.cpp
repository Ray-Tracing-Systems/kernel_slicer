#include "test_class.h"

#include "LiteMath.h"
using LiteMath::InterlockedAdd;

void SimpleTest::CalcAndAccum(uint32_t a_threadsNum, int32_t* a_out, uint32_t a_size)
{
  kernel1D_CalcAndAccum(a_threadsNum, a_out, a_size);
}

void SimpleTest::kernel1D_CalcAndAccum(uint32_t a_threadsNum, int32_t* a_out, uint32_t a_size)
{
  #pragma omp parallel for
  for(uint32_t i=0; i < a_threadsNum; i++)
  {
    uint32_t addr = i % a_size;
    InterlockedAdd(a_out[addr], int(i));
  }
}