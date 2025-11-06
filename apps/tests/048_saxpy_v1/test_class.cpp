#include "test_class.h"
#include <cstdint>
#include <vector>

void Numbers::SAXPY(const float* a_adata, 
                    const float* a_bdata, 
                    const float* a_cdata, 
                          float* a_result, unsigned int a_dataSize)
{
  std::vector<float> temp(a_dataSize);
  kernel1D_Mult(temp.data(), a_adata, a_bdata, a_dataSize);
  kernel1D_Add(a_result, temp.data(), a_cdata, a_dataSize);
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
