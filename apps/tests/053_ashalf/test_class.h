#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>

#define HALFFLOAT
#include "LiteMath.h"

using LiteMath::half;
using LiteMath::half4;
using LiteMath::uint2;
using LiteMath::float4;


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

struct LBPixel 
{
  half     color[3];
  uint16_t index;
};

class Test2D 
{
public:

  Test2D(size_t a_size);

  virtual void Run(int a_size, float4* outData1ui [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

  std::vector<LBPixel> m_testPixels;
  std::vector<half4>   m_testPixels2;

protected:

  void kernel1D_Eval(int a_size, float4* outData4f);
};

#endif