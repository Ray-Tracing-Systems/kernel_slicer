#include "test_class.h"

void SimpleTest::CalcAndAccum(uint32_t a_threadsNum, float* a_out)
{
  kernel1D_CalcAndAccum(a_threadsNum, a_out);
}

float SimpleTest::memberFunction(int i) { return float(i) + 1.0f; }

float SimpleTest::memberFunction2(float x, float y, float z) { return x*x + y*y + z*z + x*y + y*z + x*z + z + y + z + 1.0f; }

void SimpleTest::kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out)
{
  for(int i=0; i < a_threadsNum; i++)
  {
    float v1 = memberFunction2(m_valX, m_valY, 1.0f);
    float v2 = memberFunction2(m_valX + m_valY, m_valX*m_valY, 2.0f);
    float v3 = memberFunction2(m_valX + m_valY*m_valZ, m_valY, 1.0f);
    float v4 = memberFunction2((m_valX + m_valY)*m_valZ, m_valY, 1.0f);
    a_out[i] = memberFunction(i) + v1 + v2 + v3 + v4;
  }
}