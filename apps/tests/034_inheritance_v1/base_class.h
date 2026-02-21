#pragma once
#include <vector>

class Base
{
public:
  Base() { }

  virtual void Init(size_t a_size);

  virtual void Test(float* a_data [[size("a_size")]], unsigned int a_size);
  virtual void kernel1D_Test(float* a_data, unsigned int a_size);
  virtual void kernel1D_OnlyBase(float* a_data, unsigned int a_size);
  
  float dataInBaseClass;
  std::vector<float> vInBase;
  std::vector<float> vInBase2;

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
