#include "test_class.h"
#include <cmath>

#ifndef CUDA_MATH
using LiteMath::ReduceAddInit;
using LiteMath::ReduceAddComplete;
using LiteMath::ReduceAdd;
#endif

SimpleTest::SimpleTest() 
{
  m_accum.resize(5); 
  for(auto& x : m_accum) 
    x = 0;
}

void SimpleTest::CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out)
{
  ReduceAddInit(m_accum, m_accum.size());
  kernel1D_CalcAndAccum(in_data, a_threadsNum, a_out);
  ReduceAddComplete(m_accum);
  kernel1D_CopyData(a_out, m_accum.data(), uint32_t(m_accum.size()));
}

void SimpleTest::kernel1D_CalcAndAccum(const float* in_data, uint32_t a_threadsNum, float* a_out)
{
  #pragma omp parallel for
  for(int i=0; i < a_threadsNum; i++)
  {
    //float x = in_data[i];
    ReduceAdd<float, uint32_t>(m_accum, 0, 1.0f);
    ReduceAdd<float, uint32_t>(m_accum, 1, 2.0f);
    ReduceAdd<float, uint32_t>(m_accum, 2, 3.0f);
    ReduceAdd<float, uint32_t>(m_accum, 3, 4.0f);
    ReduceAdd<float, uint32_t>(m_accum, 4, 5.0f);
  }
}

void SimpleTest::kernel1D_CopyData(float* a_out, const float* a_in, uint32_t a_size)
{
  for(uint32_t i=0; i < a_size; i++)
    a_out[i] = a_in[i];
}