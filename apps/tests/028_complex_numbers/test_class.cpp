#include "test_class.h"
#include <cstdint>

using LiteMath::complex;
using LiteMath::M_PI;

static inline complex filmPhaseDiff(complex cosTheta, complex eta, float thickness, float lambda)
{
  return 4 * M_PI * eta * cosTheta * thickness / complex(lambda);
}

void TestClass::Test(BoxHit* a_data, unsigned int a_size)
{
  Cow testCow;
  testCow.moooo = 2.0f;
  kernel1D_Test(a_data, a_size, testCow);
}

void TestClass::kernel1D_Test(BoxHit* a_data, uint32_t a_size, Cow a_cow)
{
  for(uint32_t i=0; i<a_size; i++) 
  {
    complex a = complex(1.0f, 0.0f);
    complex b = complex(1.0f, -1.0f);
    complex test = filmPhaseDiff(a, b, 3.0f, 4.0f);
    a_data[i] = make_BoxHit(i, a_cow.moooo + test.re + test.im);
  }
}
