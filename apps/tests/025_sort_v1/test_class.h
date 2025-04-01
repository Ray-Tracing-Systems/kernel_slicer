#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "LiteMath.h"
using LiteMath::uint2;

class Sorter
{
public:
  Sorter(){}

  virtual void Sort(const uint2* a_data   __attribute__((size("a_size"))), unsigned int a_size,
                          uint2* a_outExc __attribute__((size("a_size"))));

  virtual void kernel1D_CopyData(const uint2* a_in, uint2* a_out, unsigned int a_size);                        

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
