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

      a_data[3] = m_badData.pos[0];
      a_data[4] = m_badData.pos[1];
      a_data[5] = m_badData.pos[2];
      a_data[6] = m_badData.pos[3];

      a_data[7] = m_badData.intensity[0];
      a_data[8] = m_badData.intensity[1];
      a_data[9] = m_badData.intensity[2];
      a_data[10] = m_badData.intensity[3];
      
      a_data[11] = m_badData.norm[0];
      a_data[12] = m_badData.norm[1];
      a_data[13] = m_badData.norm[2];
      a_data[14] = m_badData.norm[3];

      a_data[15] = m_badData.size[0];
      a_data[16] = m_badData.size[1];
      
      a_data[17] = test1;
      a_data[18] = test2;
      a_data[19] = test3;
    }
  }
}
