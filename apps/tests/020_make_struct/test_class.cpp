#include "test_class.h"
#include <cstdint>

void TestClass::Test(BoxHit* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);
}

void TestClass::kernel1D_Test(BoxHit* a_data, uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++)
  {
    //BoxHit dummy; // don't help actually
    a_data[i] = make_BoxHit(i, 1.0f);
  }
}
