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
      a_data[0] = m_badData.data[0];
      a_data[1] = m_badData.data[1];
      a_data[2] = m_badData.data[2];
      a_data[3] = m_badData.data[3];
      a_data[4] = m_badData.data[4];

      a_data[5] = m_data1.x;
      a_data[6] = m_data1.y;
      a_data[7] = m_data1.z;
      a_data[8] = m_data1.w;

      a_data[9] = m_data2.x;
      a_data[10] = m_data2.y;

      a_data[11] = m_data3;
      a_data[12] = m_data4;
      a_data[13] = m_data5;
    }
  }
}
