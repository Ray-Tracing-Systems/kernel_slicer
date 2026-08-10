#include "test_class.h"

void Test2D::kernel1D_MemSet1(const int a_size, int a_offset, int a_val, int* outData1ui)
{
  for(int i=0;i<a_size;i++)
  {
    outData1ui[a_offset + i]  = a_val;
  }
}

void Test2D::kernel2D_MemSet2(const int a_width, int a_height, int a_pitch, int a_val, int* outData1ui)
{
  for(int x=0; x < a_width; x++) 
  {
    for(int y=0; y < a_height; y++) 
    {
      outData1ui[a_pitch*y + x] = a_val;
    } 
  }
}

void Test2D::Run(const int a_size, int* outData1ui)
{
  kernel1D_MemSet1(a_size,0,1,outData1ui);
  kernel2D_MemSet2(8,8,8,2,outData1ui);
}