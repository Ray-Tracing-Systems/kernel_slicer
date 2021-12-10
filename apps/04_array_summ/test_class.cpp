#include "test_class.h"
#include <string>

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