#include "test_class.h"
#include <cstdint>

VectorTest::VectorTest(size_t a_size)
{
  m_data1.resize(a_size);
  m_data2.resize(a_size);
  for(size_t i=0;i<a_size;i++)
  {
    m_data1[i] = float(i);
    m_data2[i] = float(i)*2.0f;
  }
}

void VectorTest::Test(float* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);
}

float VectorTest::getSomeData(int index) {  return m_data2[index]; }

void VectorTest::kernel1D_Test(float* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    a_data[i] = m_data1[i] + m_data2[i] + getSomeData(i);
  }
}
