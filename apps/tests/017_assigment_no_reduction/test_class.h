#pragma once
#include <vector>
#include <iostream>
#include <fstream>

class Numbers
{
public:
  Numbers(){}

  virtual void CalcArraySumm(const double* a_data __attribute__((size("a_dataSize"))), unsigned int a_dataSize);
  void kernel1D_ArraySumm(const double* a_data, size_t a_dataSize);
  int m_flag = 0;

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
