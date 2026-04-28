#pragma once

#include "base_class.h"

class Derived : public Base
{
public:
  Derived() { }

  void Init(size_t a_size) override;

  void Test(float* a_data [[size("a_size")]], unsigned int a_size) override;
  void kernel1D_Test(float* a_data, unsigned int a_size) override;

  void CommitDeviceData() override {}                                       // will be overriden in generated class
  void GetExecutionTime(const char* a_funcName, float a_out[4]) override {} // will be overriden in generated class
  
  float dataInDerivedClass;
  std::vector<float> vInDerived;
};