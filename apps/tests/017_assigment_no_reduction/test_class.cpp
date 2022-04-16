#include "test_class.h"
#include <cstdint>

void Numbers::CalcArraySumm(const double* a_data, unsigned int a_dataSize)
{
  kernel1D_ArraySumm(a_data, a_dataSize);
}

void Numbers::kernel1D_ArraySumm(const double* a_data, size_t a_dataSize)
{
  m_flag = 0;
  for(int i=0; i<a_dataSize; i++)
  {
    double number = a_data[i];
    if(number > 1000.0)
      m_flag = 1;
  }
}
