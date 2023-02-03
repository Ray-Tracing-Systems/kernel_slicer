#include "test_class.h"
#include <cstdint>

void TestClass::Test(BoxHit* a_data, unsigned int a_size)
{
  Cow testCow;
  testCow.moooo = 2.0f;
  kernel1D_Test(a_data, a_size, testCow);
}

void TestClass::kernel1D_Test(BoxHit* a_data, uint32_t a_size, Cow a_cow)
{
  for(uint32_t i=0; i<a_size; i++)
    a_data[i] = make_BoxHit(i, a_cow.moooo);
}
