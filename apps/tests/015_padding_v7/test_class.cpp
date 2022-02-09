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
      a_data[0] = m_data1[0][0] + m_data3[0].depth;
      a_data[1] = m_data1[0][1];
      a_data[2] = m_data1[0][2];

      a_data[3] = m_data1[1][0];
      a_data[4] = m_data1[1][1];
      a_data[5] = m_data1[1][2];
      
      a_data[6] = m_data1[2][0];
      a_data[7] = m_data1[2][1];
      a_data[8] = m_data1[2][2];

      a_data[9+0] = m_data2[0].data[0];
      a_data[9+1] = m_data2[0].data[1];
      a_data[9+2] = m_data2[0].data[2];

      a_data[9+3] = m_data2[1].data[0];
      a_data[9+4] = m_data2[1].data[1];
      a_data[9+5] = m_data2[1].data[2];
      
      a_data[9+6] = m_data2[2].data[0];
      a_data[9+7] = m_data2[2].data[1];
      a_data[9+8] = m_data2[2].data[2];
    }
  }
}
