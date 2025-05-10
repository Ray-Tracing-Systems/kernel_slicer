#pragma once
#include "LiteMath.h"
#include <cstdint>


class SimpleTest
{
public:
  SimpleTest() : m_valX(1.0f), m_valY(2.0f), m_valZ(3.0f) {}
  
  float memberFunction(int i);
  float memberFunction2(float x, float y, float z);
  
  virtual void CalcAndAccum(uint32_t a_threadsNum, float* a_out [[size("a_threadsNum")]]);
  virtual void kernel1D_CalcAndAccum(uint32_t a_threadsNum, float* a_out);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  float m_valX;
  float m_valY;
  float m_valZ;
};
