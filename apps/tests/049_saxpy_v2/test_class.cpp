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
  std::vector<float> temp_c1(sizeLocal);
  std::vector<float> temp_c2(sizeLocal);
  
  kernel1D_SetConst(temp_c1.data(), 1.0f, temp_c1.size());
  kernel1D_SetConst(temp_c2.data(), 2.0f, temp_c2.size());

  kernel1D_Mult(temp.data(), a_adata, a_bdata, sizeLocal);
  kernel1D_Add(a_result, temp.data(), a_cdata, sizeLocal);

  kernel1D_Add(a_result, a_result, temp_c1.data(), sizeLocal);
  kernel1D_Add(a_result, a_result, temp_c2.data(), sizeLocal);
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

void Numbers::kernel1D_SetConst(float* a_res, float a_val, size_t a_dataSize)
{
  for(int i=0; i<a_dataSize; i++) {
    a_res[i] = a_val;
  }
}
