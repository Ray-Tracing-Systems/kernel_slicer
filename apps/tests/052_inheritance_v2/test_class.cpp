#include "test_class.h"
#include <cstdint>

void Base::Init(size_t a_size)
{
  dataInBaseClass = 2.0f;
  
  vInBase.resize(a_size); 
  vInBase2.resize(a_size);
  vInBase3.resize(a_size);

   for(size_t i=0;i<a_size;i++) {
    vInBase [i] = i*2.0f;
    vInBase2[i] = i*4.0f;
    vInBase3[i].x = float(i);
  }
}

void Base::Test(float* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size); // Base class
}

void Base::kernel1D_Test(float* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    a_data[i] = dataInBaseClass*vInBase[i] + 1.0f;
  }
}

float f0() { return 1.0f;}

float Base::f1() { return 1.0f + f0() + C1; };
float Base::f2() { return 1.0f + f1() + vInBase3[0].x; };
float Base::f3() { return 1.0f + f2(); };

void Base::kernel1D_OnlyBase(float* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
    a_data[i] += dataInBaseClass + vInBase2[i] + f3();
}

void Base::Test_OnlyBase(float* a_data, unsigned int a_size)
{
  kernel1D_OnlyBase(a_data, a_size);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void Derived::Init(size_t a_size)
{
  Base::Init(a_size);
  dataInDerivedClass = 2.0f;
  vInDerived.resize(a_size);
  for(size_t i=0;i<a_size;i++)
    vInDerived[i] = i*2.0f;
}

void Derived::Test(float* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);     // called from Derived class
  kernel1D_OnlyBase(a_data, a_size); // called from Base class
}

void Derived::kernel1D_Test(float* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    a_data[i] = dataInBaseClass*vInBase[i] + dataInDerivedClass*vInDerived[i]; // + f3();
  }
}
