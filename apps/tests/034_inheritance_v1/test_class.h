#pragma once
#include <vector>
#include <iostream>
#include <fstream>

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