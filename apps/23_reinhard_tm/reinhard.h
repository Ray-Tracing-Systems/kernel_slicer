#ifndef TEST_REINHARD_H
#define TEST_REINHARD_H

#include <vector>
#include <iostream>
#include <fstream>

class ReinhardTM 
{
public:

  ReinhardTM(){}

  virtual void Run(int w, int h, const float* inData4f __attribute__((size("w*h*4"))), uint32_t* outData __attribute__((size("w*h"))));

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    
  
  float getWhitePoint() { return m_whitePoint; }

protected:

  void kernel1D_findMax(int size, const float* inData4f);
  void kernel2D_process(int w, int h, const float* inData4f, uint32_t* outData);

  float m_whitePoint = 0;
  int m_width, m_height;

};

#endif