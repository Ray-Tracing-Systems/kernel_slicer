#include "test_class.h"

using LiteMath::dot;

SimpleTest::SimpleTest()
{
  m_mat1 = float4x4(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16);
  m_mat2 = float4x4(1,-1,1,-1, -1,1,-1,1, -1,1,1,-1, 1,-1,-1,1);
}

void SimpleTest::CalcAndAccum(uint32_t a_threadsNum, float* a_out, uint32_t a_size)
{
  kernel1D_CalcAndAccum(a_threadsNum, a_out, a_size);
}

void SimpleTest::kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out, uint32_t a_size)
{
  for(int i=0; i < a_threadsNum; i++)
  {
    //float4x4 m  = m_mat1*m_mat2; // BUG!!!!
    float4x4 m1  = m_mat1;
    float4x4 m2  = m_mat2;
    float4x4 m  = m1*m2;
    float4 val  = float4(i,i*0.5f,i*2.0f,-i); 
    float4 valT = m*val;
    a_out[i]    = dot(valT,valT);
  }
}