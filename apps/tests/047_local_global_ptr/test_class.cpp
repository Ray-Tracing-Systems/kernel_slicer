#include "test_class.h"

static inline int read_data(const int* a_data, int offset, int a_id)
{
  return a_data[offset + a_id];
}

static inline int read_data2(const int* a_data, int a_id)
{
  return read_data(a_data, 0, a_id) + read_data(a_data, 1, a_id);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

TestVecDataAccessFromMember::TestVecDataAccessFromMember(size_t a_size)
{
  m_vec.resize(a_size);
  for(size_t i=0;i<m_vec.size();i++)
    m_vec[i] = int(i) + 10;
}

int TestVecDataAccessFromMember::getMemberData(int a_id)
{
  return read_data(m_vec.data(), 2, a_id - 2) + read_data2(m_vec.data(), 5);
}

void TestVecDataAccessFromMember::kernel1D_Run(const int a_size, int* outData1ui)
{
  for(int i=0;i<a_size;i++)
  {
    const int val0 = read_data(m_vec.data(), 0, i);
    const int val1 = read_data2(m_vec.data(), 10);
    outData1ui[i]  = getMemberData(i) + val0 + val1;
  }
}

void TestVecDataAccessFromMember::Run(const int a_size, int* outData1ui)
{
  kernel1D_Run(a_size, outData1ui);
}