#include "test_class.h"
#include <cstdint>
#include <vector>

void Numbers::SAXPY(const float* a_adata, 
                    const float* a_bdata, 
                    const float* a_cdata, 
                          float* a_result, unsigned int a_dataSize)
{
  unsigned int sizeLocal = a_dataSize;
  std::vector<float> temp(sizeLocal);
  
  kernel1D_Mult(temp.data(), a_adata, a_bdata, sizeLocal);
  kernel1D_Add(a_result, temp.data(), a_cdata, sizeLocal);
}
                          

void Numbers::kernel1D_Mult(float* a_res, 
                            const float* a_input1, 
                            const float* a_input2, size_t a_dataSize) {
  for(int i=0; i<a_dataSize; i++) {
    a_res[i] = a_input1[i] * a_input2[i];
  }
}

void Numbers::kernel1D_Add(float* a_res, 
                           const float* a_input1, 
                           const float* a_input2, size_t a_dataSize) {
  for(int i=0; i<a_dataSize; i++) {
    a_res[i] = a_input1[i] + a_input2[i];
  }
}              

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Numbers::SAXPY2Block(const float* a_adata, const float* a_bdata, const float* a_cdata, float* a_result, unsigned int a_size, unsigned int a_passesNum)
{
  for(int tid=0; tid < a_size; tid++)
    SAXPY2(a_adata, a_bdata, a_cdata, a_result, tid);
}

void Numbers::SAXPY2(const float* a_adata, const float* a_bdata, const float* a_cdata, float* a_result, unsigned int tid)
{
  float temp;
  kernel_Mult2(&temp, a_adata, a_bdata, tid);
  kernel_Add2(a_result, &temp, a_cdata, tid);
}

void Numbers::kernel_Mult2(float* a_res, const float* a_input1, const float* a_input2, unsigned int tid)
{
  *a_res = a_input1[tid] * a_input2[tid];
}

void Numbers::kernel_Add2 (float* a_res, const float* a_input1, const float* a_input2, unsigned int tid)
{
  a_res[tid] = *a_input1 + a_input2[tid];
}
