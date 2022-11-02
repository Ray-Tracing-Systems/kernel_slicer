#ifndef TEST_REINHARD_H
#define TEST_REINHARD_H

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif

class Mandelbrot 
{
public:

  Mandelbrot(){}

  virtual void Fractal(int w, int h, uint32_t* outData __attribute__((size("w*h"))));

  virtual void CommitDeviceData() {}                                                           // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_time; } // will be overriden in generated class    

protected:

  virtual void kernel2D_process(int w, int h, uint32_t* outData);
  float m_time;
};

#endif