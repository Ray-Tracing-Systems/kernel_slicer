#pragma once
#include <vector>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <fstream>

#include "LiteMath.h"

struct TestData
{
  int   key;
  float val;
  float dummy1;
  float dummy2;
};

class Sorter
{
public:
  Sorter(){}

  virtual void Reserve(size_t n) { m_testData.resize(n); }
  virtual void Sort(const TestData* a_data   __attribute__((size("a_size"))), unsigned int a_size,
                          TestData* a_outExc __attribute__((size("a_size"))));

  virtual void kernel1D_CopyData(const TestData* a_in, TestData* a_out, unsigned int a_size);                        

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  std::vector<TestData> m_testData;
};
