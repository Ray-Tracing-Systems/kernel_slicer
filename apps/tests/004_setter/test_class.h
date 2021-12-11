#pragma once
#include <vector>
#include <iostream>
#include <fstream>

struct MyInOut
{
  int* summ;
  int* product;
  int* reduction;
  unsigned someSize;
};

class ArrayProcess
{
public:
  ArrayProcess(){}
  
  [[kslicer::setter]] void SetOutput(MyInOut a_out);

  virtual void ProcessArrays(const int* a_data1 __attribute__((size("a_dataSize"))), 
                             const int* a_data2 __attribute__((size("a_dataSize"))), unsigned a_dataSize);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void ReserveTestData(size_t n) { testData.reserve(n); }

 protected: 
  void kernel1D_ArrayProc(const int* a_data1, const int* a_data2, unsigned a_dataSize);

  int m_summ;
  int m_minVal;
  int m_maxVal;
  MyInOut m_out;

  float m_testArray[4];

  std::vector<float> testData;
};
