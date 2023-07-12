#include "test_class.h"

static inline int read_data(const int* a_data, int a_id)
{
  return a_data[a_id];
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
  return read_data(m_vec.data(), a_id);
}

void TestVecDataAccessFromMember::kernel1D_Run(const int a_size, int* outData1ui)
{
  for(int i=0;i<a_size;i++)
  {
    outData1ui[i] = getMemberData(i);
  }
}

void TestVecDataAccessFromMember::Run(const int a_size, int* outData1ui)
{
  kernel1D_Run(a_size, outData1ui);
}