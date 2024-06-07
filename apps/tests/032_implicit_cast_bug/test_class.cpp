#include "test_class.h"


void TestClass::Test(float* a_data, unsigned int a_size)
{
  Variable var;
  var.offset = 0;
  kernel1D_fill(a_data, a_size, var, 1.0f);
}

void TestClass::kernel1D_fill(float *data, unsigned steps, Variable A, float val)
{
  for (unsigned i = 0; i < steps; i++) {
    data[A.offset + i] = val;
  }
}