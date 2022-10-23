#include "test_class.h"
#include <string>
#include <chrono>

void Numbers::CalcArraySumm(const int* a_data, unsigned int a_dataSize)
{
  auto before = std::chrono::high_resolution_clock::now();
  kernel1D_ArraySumm(a_data, a_dataSize);
  m_executionTime = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - before).count()/1000.f;
}

void Numbers::kernel1D_ArraySumm(const int* a_data, size_t a_dataSize)
{
  m_summ = 0;
  for(int i=0; i<a_dataSize; i++)
  {
    if(a_data[i] > 0)
      m_summ += a_data[i];
  }
}