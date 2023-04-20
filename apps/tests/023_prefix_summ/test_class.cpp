#include "test_class.h"

#include <algorithm>
#include <numeric>

void PrefSummTest::PrefixSumm(const int* a_data, unsigned int a_size, int* a_outExc, int* a_outInc)
{
  kernel1D_Test(a_data, 1); // need to detect 'PrefixSumm' as control function
  std::exclusive_scan(a_data, a_data + a_size, a_outExc, 0);
  std::inclusive_scan(a_data, a_data + a_size, a_outInc, std::plus<int>(), 0);
  //memcpy(a_outExc, a_data, sizeof(int)*a_size);
}

void PrefSummTest::kernel1D_Test(const int* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    
  }
}
