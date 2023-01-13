#pragma once
#include <vector>
#include <iostream>
#include <fstream>

class Numbers
{
public:
  Numbers(){}
  virtual void CalcArraySumm(const int* a_data __attribute__((size("a_dataSize"))), unsigned int a_dataSize);
  virtual void kernel1D_ArraySumm(const int* a_data, size_t a_dataSize);
  float m_summ;

  virtual void CommitDeviceData() {}                                                                    // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) { a_out[0] = m_executionTime; } // will be overriden in generated class
  float m_executionTime = 0.0f;
};
