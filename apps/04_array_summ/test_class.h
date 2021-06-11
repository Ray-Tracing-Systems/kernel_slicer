#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include "OpenCLMath.h"

class Numbers
{
public:
 
  void CalcArraySumm(const int* a_data, uint a_dataSize);
  void kernel1D_ArraySumm(const int* a_data, size_t a_dataSize);
  int m_summ;
};
