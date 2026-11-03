#include "test_class.h"
#include <cmath>

static constexpr uint32_t INVALID_VALUE = (uint32_t)-1;

static inline float scaled(float x, float scale = 2.0f, uint32_t add = 3)
{
  return x*scale + float(add);
}

static inline float half_of(float x)
{
  static constexpr float K = 0.5f;
  return x*K;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

TestCppConstructs::TestCppConstructs(size_t a_size)
{
  m_data.resize(a_size);
  for(size_t i=0;i<m_data.size();i++)
    m_data[i] = float(i % 10) + 1.0f;
}

void TestCppConstructs::kernel1D_Run(const int a_size, float* outData)
{
  for(int i=0;i<a_size;i++)
  {
    const float    x   = m_data[i];
    const uint32_t odd = uint32_t(i) & 1u;
    const uint32_t id  = (i % 5 == 0) ? INVALID_VALUE : uint32_t(i);

    float res = odd ? x : -x;                                                 // uint --> bool
    if(odd)
      res += 1.0f;
    res += scaled(x) + 10.0f*scaled(x, 3.0f) + 100.0f*scaled(x, 3.0f, 4u);    // default arguments
    res += half_of(x);                                                        // local 'static constexpr float'
    res += (x < INFINITY) ? 1000.0f : 0.0f;
    res += (std::nextafter(x, INFINITY) - x)*1048576.0f;                      // one ulp of x, power of 2
    res += (id == INVALID_VALUE) ? 5.0f : 0.0f;                               // '(uint32_t)-1'
    res += m_flagA ? 10000.0f : 0.0f;                                         // bool members in UBO
    res += m_flagB ? 20000.0f : 0.0f;
    res += m_flagC ? 40000.0f : 0.0f;
    outData[i] = res;
  }
}

void TestCppConstructs::Run(const int a_size, float* outData)
{
  kernel1D_Run(a_size, outData);
}
