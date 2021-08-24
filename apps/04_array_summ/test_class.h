#pragma once
#include <vector>
#include <iostream>
#include <fstream>

class Numbers
{
public:
 
  void CalcArraySumm(const int* a_data, unsigned int a_dataSize);
  void kernel1D_ArraySumm(const int* a_data, size_t a_dataSize);
  int m_summ;
};
