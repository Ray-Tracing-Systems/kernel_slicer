#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>

#include "LiteMath.h"

using LiteMath::float4;

class BoxMinMax
{
public:

  BoxMinMax(){}
  virtual void ProcessPoints(const float4* a_inData  __attribute__((size("a_dataSize"))), size_t a_dataSize,
                                   float4* a_outData __attribute__((size("2"))));

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
  
  void kernel1D_FindBoundingBox(const float4* a_inData, uint32_t a_dataSize, float4* a_outData);
 
  float4 m_boxMin;
  float4 m_boxMax;
};

#endif