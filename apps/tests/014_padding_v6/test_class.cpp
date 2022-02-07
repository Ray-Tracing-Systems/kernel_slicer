#include "test_class.h"
#include <cstdint>

void Padding::Test(float* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);
}

void Padding::kernel1D_Test(float* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    if(i == 0)
    {
      a_data[0] = matrix[0][0];
      a_data[1] = matrix[1][1];
      a_data[2] = matrix[2][2];

      a_data[3] = m_badData.intensity[0];
      a_data[4] = m_badData.intensity[1];
      a_data[5] = m_badData.intensity[2];
      
      a_data[6] = m_badData.value;

      a_data[7] = test1;
      a_data[8] = test2;
      a_data[9] = test3;
    }
  }
}
