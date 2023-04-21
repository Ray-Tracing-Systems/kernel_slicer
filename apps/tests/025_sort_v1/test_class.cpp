#include "test_class.h"

#include <algorithm>
#include <numeric>

void Sorter::Sort(const uint2* a_data, unsigned int a_size, uint2* a_outExc)
{
  kernel1D_CopyData(a_data, a_outExc, a_size); 
  std::sort(a_outExc, a_outExc + a_size, [](uint2 a, uint2 b) { return a.x < b.x; });
}

void Sorter::kernel1D_CopyData(const uint2* a_in, uint2* a_out, unsigned int a_size)  
{
  for(int i=0; i<a_size; i++)
  {
    a_out[i] = a_in[i];
  }
}
