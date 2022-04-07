#ifndef TEST_REINHARD_H
#define TEST_REINHARD_H

#include <vector>
#include <iostream>
#include <fstream>


class ReinhardTM 
{
public:

  ReinhardTM(int a_w, int a_h) : m_width(a_w), m_height(a_h) {}

  virtual void Run(int w, int h, const float* inData4f __attribute__((size("w", "h", "4"))), unsigned int* outData __attribute__((size("w", "h"))));

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel1D_findMax(int size, const float* inData4f);
  void kernel2D_doToneMapping(int w, int h, const float* inData4f, unsigned int* outData);

  int m_width, m_height;
  float m_whitePoint = 0.0f;
};

#endif