#pragma once
#include <vector>
#include <iostream>
#include <fstream>

class Numbers
{
public:
  Numbers(){}

  virtual void SAXPY(const float* a_adata  [[size("a_dataSize")]], 
                     const float* a_bdata  [[size("a_dataSize")]],
                     const float* a_cdata  [[size("a_dataSize")]],
                           float* a_result [[size("a_dataSize")]], unsigned int a_dataSize);

  void kernel1D_Mult(float* a_res, 
                     const float* a_input1, 
                     const float* a_input2, size_t a_dataSize);

  void kernel1D_Add(float* a_res, 
                    const float* a_input1, 
                    const float* a_input2, size_t a_dataSize);

  void kernel1D_SetConst(float* a_res, float a_val, size_t a_dataSize);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
  //virtual size_t GetTempBufferSize() const { return size_t(2048*2048*16); }
};
