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
  
  [[kslicer::setter]] void SetOutput(MyInOut a_out);
  void ProcessArrays(const int* a_data1, const int* a_data2, unsigned a_dataSize);

  void ReserveTestData(size_t n) { testData.reserve(n); }

 protected: 
  void kernel1D_ArrayProc(const int* a_data1, const int* a_data2, unsigned a_dataSize);
  void set_product(int i);

  int m_summ;
  int m_minVal;
  int m_maxVal;
  int m_testVal = 222;
  MyInOut m_out;

  std::vector<float> testData;
};
