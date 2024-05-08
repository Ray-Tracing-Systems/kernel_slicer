#include "test_class.h"

float4x4 TestMatFunc()
{
  float4x4 m;
  m[0][0] = 1.0f;
  m[0][1] = 2.0f;
  m[0][2] = 3.0f;
  m[0][3] = 4.0f;

  m[1][0] = 5.0f;
  m[1][1] = 6.0f;
  m[1][2] = 7.0f;
  m[1][3] = 8.0f;

  m[2][0] = 9.0f;
  m[2][1] = 10.0f;
  m[2][2] = 11.0f;
  m[2][3] = 12.0f;

  m[3][0] = 13.0f;
  m[3][1] = 14.0f;
  m[3][2] = 15.0f;
  m[3][3] = 16.0f;

  return m;
}

void TestClass::Test(float4* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);
}

void TestClass::kernel1D_Test(float4* a_data, uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++) 
  {
    float4 testInput = float4(1.0f,2.0f,3.0f,4.0f);

    float4x4 testMatrix;
    testMatrix[0][0] = 1.0f;
    testMatrix[0][1] = 2.0f;
    testMatrix[0][2] = 3.0f;
    testMatrix[0][3] = 4.0f;

    testMatrix[1][0] = 5.0f;
    testMatrix[1][1] = 6.0f;
    testMatrix[1][2] = 7.0f;
    testMatrix[1][3] = 8.0f;

    testMatrix[2][0] = 9.0f;
    testMatrix[2][1] = 10.0f;
    testMatrix[2][2] = 11.0f;
    testMatrix[2][3] = 12.0f;

    testMatrix[3][0] = 13.0f;
    testMatrix[3][1] = 14.0f;
    testMatrix[3][2] = 15.0f;
    testMatrix[3][3] = 16.0f;

    if(i == 0)
      a_data[i] = m_testMatrix*testInput;
    else if(i == 1)
      a_data[i] = m_testMatrices[i]*testInput;
    else if (i == 2) 
      a_data[i] = testMatrix*testInput;
    else 
      a_data[i] = TestMatFunc()*testInput;
  }
}
