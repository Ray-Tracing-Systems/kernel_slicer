#include "test_class.h"

#include <algorithm>
#include <numeric>

void Sorter::Sort(const TestData* a_data, unsigned int a_size, TestData* a_outExc)
{
  kernel1D_CopyData(a_data, m_testData.data(), a_size); 
  std::sort(m_testData.begin(), m_testData.end(), [](TestData a, TestData b) { return a.key < b.key; });
  kernel1D_CopyData(m_testData.data(), a_outExc, a_size); 
}

void Sorter::kernel1D_CopyData(const TestData* a_in, TestData* a_out, unsigned int a_size)  
{
  for(int i=0; i<a_size; i++)
  {
    a_out[i] = a_in[i];
  }
}
