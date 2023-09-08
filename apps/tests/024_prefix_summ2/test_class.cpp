#include "test_class.h"

#include <algorithm>
#include <numeric>
#include <cstring>

void PrefSummTest::PrefixSumm(const int* a_data, unsigned int a_size, int* a_outExc)
{
  kernel1D_CopyData(a_data, a_size); // need to detect 'PrefixSumm' as control function
  std::exclusive_scan(m_internalData.begin(), m_internalData.end(), m_internalData.begin(), 0);
  memcpy(a_outExc, m_internalData.data(), sizeof(int)*a_size);
}

void PrefSummTest::kernel1D_CopyData(const int* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    m_internalData[i] = uint32_t(a_data[i]);
  }
}
