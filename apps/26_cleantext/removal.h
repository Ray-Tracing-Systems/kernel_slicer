#ifndef TEST_REINHARD_H
#define TEST_REINHARD_H

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif

class TextRemoval
{
public:
  TextRemoval(){}

  virtual void Reserve(int w, int h);
  virtual void Run(int w, int h, const uint32_t* inData __attribute__((size("w*h"))), uint32_t* outData __attribute__((size("w*h"))));
 
  virtual void CommitDeviceData(){}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]){ a_out[0] = m_timeEx;}

protected:

  virtual void kernel2D_findBadPixels(int w, int h, const uint32_t* inData);
  virtual void kernel2D_emplaceBadPixels(int w, int h, const uint32_t* inData, uint32_t* outData);


  float m_timeEx;
  std::vector<int> m_badPixels;
  std::vector<int> m_mask;
  int m_width, m_height;
};

#endif