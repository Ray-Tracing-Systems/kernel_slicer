#include "test_class.h"

#include "LiteMath.h"
using LiteMath::float2;
using LiteMath::complex;

void SimpleTest::Test(float* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);
}

void SimpleTest::kernel1D_Test(float* a_data, unsigned int a_size)
{
  for(int i=0;i<a_size; i++)
  {
    if(i == 0)
    {
      float2 v1 = float2{1.0f,2.0f};
      float2 v2 = {3.0f,4.0f};
      float2 v3{5.0f,6.0f};
      float2 v4 = float2(7.0f,8.0f);
      float2 v5(9.0f,10.0f);
      float2 v6;
      v6.x = 11.0f;
      v6.y = 12.0f;
      float2 v7 = float2(v6); 

      complex c1 = complex{15.0f,16.0f};
      complex c2 = {15.0f,16.0f};
      complex c3{13.0f,14.0f};
      complex c4 = complex(13.0f,14.0f);
      complex c5(13.0f,14.0f);
      complex c6;
      c6.re = 15.0f;
      c6.im = 16.0f;

      a_data[0] = v1.x;
      a_data[1] = v1.y;
      a_data[2] = v2.x + v2.y;

      a_data[3] = v3[0];
      a_data[4] = v3[1];
      a_data[5] = v4[0] + v4[1];
      
      a_data[6] = v5.x;
      a_data[7] = v5.y;
      a_data[8] = v6.x + v6.y;

      a_data[9+0] = v7.x;
      a_data[9+1] = v7.y;
      a_data[9+2] = c1.re + c1.im;

      a_data[9+3] = c2.re + c2.im;
      a_data[9+4] = c3.re + c3.im;
      a_data[9+5] = c4.re + c4.im;
      
      a_data[9+6] = c5.re + c5.im;
      a_data[9+7] = c6.re + c6.im;
      a_data[9+8] = 0.0f;

      a_data[17]  = 10.0f;
      a_data[18]  = 20.0f;
      a_data[19]  = 30.0f; 
    }
  }
}
