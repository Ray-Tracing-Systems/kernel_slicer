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
    a_data[i*8+0] = m_data[i].boxMin.x;
    a_data[i*8+1] = m_data[i].boxMin.y;
    a_data[i*8+2] = m_data[i].boxMin.z;
    a_data[i*8+3] = float(m_data[i].leftOffset);

    a_data[i*8+4] = m_data[i].boxMax.x;
    a_data[i*8+5] = m_data[i].boxMax.y;
    a_data[i*8+6] = m_data[i].boxMax.z;
    a_data[i*8+7] = float(m_data[i].escapeIndex);
  }
}
