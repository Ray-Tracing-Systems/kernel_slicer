#ifndef TEST_REINHARD_H
#define TEST_REINHARD_H

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif

class ReinhardTM 
{
public:

  ReinhardTM(){}

  virtual void Run(int w, int h, const float4* inData4f __attribute__((size("w*h"))), uint32_t* outData __attribute__((size("w*h"))));

  virtual void CommitDeviceData() {}                                                          // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; } // will be overriden in generated class    
  
  float getWhitePoint() { return m_whitePoint; }

protected:

  virtual void kernel1D_findMax(int size, const float4* inData4f);
  virtual void kernel2D_process(int w, int h, const float4* inData4f, uint32_t* outData);

  float m_whitePoint = 0;
  float m_time;
  int m_width, m_height;

};

#endif