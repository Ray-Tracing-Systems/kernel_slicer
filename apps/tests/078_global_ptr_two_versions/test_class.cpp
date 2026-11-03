#include "test_class.h"

static inline float twice(const float* a_val)
{
  return 2.0f*(*a_val);
}

static inline float four_times(const float* a_val)
{
  return twice(a_val) + twice(a_val);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

TestGlobalPtrTwoVersions::TestGlobalPtrTwoVersions(size_t a_size)
{
  m_params.resize(16);
  for(size_t i=0;i<m_params.size();i++)
    m_params[i] = float(i % 10);
}

void TestGlobalPtrTwoVersions::kernel1D_Run(const int a_size, float* outData)
{
  for(int i=0;i<a_size;i++)
  {
    const float local = float(i % 7);
    float res = twice(&local) + 10.0f*four_times(&local);                     // original versions
    res += 100.0f*twice(m_params.data() + (i % 5)) + 1000.0f*four_times(&m_params[i % 3]); // copies for 'm_params'
    outData[i] = res;
  }
}

void TestGlobalPtrTwoVersions::Run(const int a_size, float* outData)
{
  kernel1D_Run(a_size, outData);
}
