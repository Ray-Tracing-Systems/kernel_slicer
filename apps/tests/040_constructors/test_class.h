#pragma once

class SimpleTest
{
public:
  SimpleTest()
  { 
    
  }

  virtual void Test(float* a_data [[size("a_size")]], unsigned int a_size);
  void kernel1D_Test(float* a_data, unsigned int a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class
};
