#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL

#include <vector>
#include <iostream>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class ToneMapping 
{
public:

  ToneMapping() { m_gammaInv = 1.0f / 2.2f;}
  void SetMaxImageSize(int w, int h);

  // Base on IPT color space by Ebner and Fairchild (1998).
  virtual void IPTcompress(int w, int h, const float4* inData4f __attribute__((size("w", "h"))), unsigned int* outData1ui __attribute__((size("w", "h"))));

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel1D_IPTcompress(int a_size, const float4* inData4f, unsigned int* outData1ui);
  
  int   m_width;
  int   m_height;
  float m_gammaInv;
};

#endif