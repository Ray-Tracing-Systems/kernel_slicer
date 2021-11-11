#include "test_class.h"
#include <cstdint>

void Numbers::CalcArraySumm(const int* a_data, unsigned int a_dataSize)
{
  kernel1D_ArraySumm(a_data, a_dataSize);
}

void Numbers::kernel1D_ArraySumm(const int* a_data, size_t a_dataSize)
{
  m_summ = 0;
  for(int i=0; i<a_dataSize; i++)
  {
    int number = a_data[i];
    if(number > 0)
      m_summ += number;
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int32_t array_summ_cpu(const std::vector<int32_t>& array)
{
  Numbers filter;
  auto start = std::chrono::high_resolution_clock::now();
  filter.CalcArraySumm(array.data(), uint32_t(array.size()));
  auto stop = std::chrono::high_resolution_clock::now();
  auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
  std::cout << "[cpu]: " << ms << " ms for CalcArraySumm " << std::endl;
  return filter.m_summ;
}
