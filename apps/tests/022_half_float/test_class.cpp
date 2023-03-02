#include "test_class.h"
#include <cstdint>

void Padding::Test(float* a_data, unsigned int a_size)
{
  kernel1D_Test(a_data, a_size);
}

void Padding::kernel1D_Test(float* a_data, unsigned int a_size)
{
  for(int i=0; i<a_size; i++)
  {
    const TestBox box = m_data[i];
    const float4 f1   = float4(box.b1);
    const float4 f2   = float4(box.b2);
    const float4 f3   = float4(box.b3); 

    a_data[i*14+0] = f1.x;
    a_data[i*14+1] = f1.y;
    a_data[i*14+2] = f1.z;
    a_data[i*14+3] = f1.w;

    a_data[i*14+4] = f2.x;
    a_data[i*14+5] = f2.y;
    a_data[i*14+6] = f2.z;
    a_data[i*14+7] = f2.w;

    a_data[i*14+8]  = f3.x;
    a_data[i*14+9]  = f3.y;
    a_data[i*14+10] = f3.z;
    a_data[i*14+11] = f3.w;

    a_data[i*14+12] = float(box.offs1);
    a_data[i*14+13] = float(box.offs2);
  }
}
