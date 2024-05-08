#pragma once
#include <cstdint>

#include <vector>
#include <iostream>
#include <fstream>

struct Variable
{
  unsigned Dim;
  unsigned offset;
  unsigned total_size;
  unsigned sizes[8]; //MAX_DIM
};

class TestClass
{
public:
  TestClass(){}

  virtual void Test(float* a_data [[size("a_size")]], unsigned int a_size);
  void kernel1D_fill(float *data, unsigned steps, Variable A, float val);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
