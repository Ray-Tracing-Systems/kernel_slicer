#include "test_class.h"

using LiteMath::make_float4x4_from_rows;
using LiteMath::make_float4x4_from_cols;

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
  
    ///////////////////////////////////////////////////////////
    float4x4 testMatrixRow = make_float4x4_from_rows(float4(1,2,3,4), 
                                                     float4(5,6,7,8),
                                                     float4(9,10,11,12),
                                                     float4(13,14,15,16));

    float4x4 testMatrixCol = make_float4x4_from_cols(float4(1,5,9, 13), 
                                                     float4(2,6,10,14),
                                                     float4(3,7,11,15),
                                                     float4(4,8,12,16));

    float4x4 testMatrixRow2;
    testMatrixRow2.set_row(0, float4(1,2,3,4));
    testMatrixRow2.set_row(1, float4(5,6,7,8));
    testMatrixRow2.set_row(2, float4(9,10,11,12));
    testMatrixRow2.set_row(3, float4(13,14,15,16));

    float4x4 testMatrixCol2;
    testMatrixCol2.set_col(0, float4(1,5,9, 13));
    testMatrixCol2.set_col(1, float4(2,6,10,14));
    testMatrixCol2.set_col(2, float4(3,7,11,15));
    testMatrixCol2.set_col(3, float4(4,8,12,16));

    if(i == 0)
      a_data[i] = m_testMatrix*testInput;
    else if(i == 1)
      a_data[i] = m_testMatrices[i]*testInput;
    else if (i == 2) 
      a_data[i] = testMatrix*testInput;
    else if (i == 3) 
      a_data[i] = TestMatFunc()*testInput;
    else if (i == 4)
      a_data[i] = testMatrixRow*testInput;
    else if (i == 5)
      a_data[i] = testMatrixCol*testInput;
    else if (i == 6)
      a_data[i] = testMatrixRow2*testInput;
    else if (i == 7)
      a_data[i] = testMatrixCol2*testInput;
    else 
      a_data[i] = float4(0,0,0,0);
  }
}
