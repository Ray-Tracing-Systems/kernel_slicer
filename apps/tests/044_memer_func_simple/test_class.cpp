#include "test_class.h"

SimpleTest::SimpleTest() {}

void SimpleTest::CalcAndAccum(uint32_t a_threadsNum, float* a_out)
{
  kernel1D_CalcAndAccum(a_threadsNum, a_out);
}

float SimpleTest::memberFunction(int i) { return float(i) + 1.0f; }

void SimpleTest::kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out)
{
  for(int i=0; i < a_threadsNum; i++)
  {
    a_out[i] = memberFunction(i);
  }
}