#include "test_class.h"

float TestClass::fSumm(int tid)
{
  [[threadlocal]] float tempData2[8];
    
  for(int j=0;j<8;j++)
    tempData2[j] = float(j*tid + j);

  float summ = 0.0f;
  for(int j=0;j<8;j++)
    summ += tempData2[j];
  
  return summ;
}

void TestClass::Test(float* a_data, int a_size)
{
  kernel1D_Test(a_data, a_size);
}

void TestClass::kernel1D_Test(float* a_data, int a_size)
{
  for(int i=0; i<a_size; i++) 
  {
    [[threadlocal]] float tempData[16];
    float2 tempData2[4];
    
    for(int j=0;j<16;j++)
      tempData[j] = float(j*i + j);

    float summ = 0.0f;
    for(int j=0;j<16;j++)
      summ += tempData[j];

    a_data[i] = summ + fSumm(i);
  }
}
