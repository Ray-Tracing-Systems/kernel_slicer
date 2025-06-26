#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TestVecDataAccessFromMember 
{
public:

  TestVecDataAccessFromMember(size_t a_size);  

  virtual void Run(const int a_size, int* outData1ui [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel1D_Run(const int a_size, int* outData1ui);
  int  getMemberData(int a_id);
  std::vector<int> m_vec;
};

#endif