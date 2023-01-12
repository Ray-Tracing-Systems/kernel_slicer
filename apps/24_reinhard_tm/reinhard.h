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
  
  virtual void Run(int w, int h, const float* inData __attribute__((size("w*h*4"))), uint32_t* outData __attribute__((size("w*h"))));

  virtual void CommitDeviceData(){}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]){ a_out[0] = m_time;}

  float getWhitePoint() const { return whitePoint; }

protected:

  virtual void kernel1D_finMax(const float* inData, int size);
  virtual void kernel2D_process(int w, int h, const float* inData, uint32_t* outData);

  float whitePoint;
  float m_time;
};


#endif