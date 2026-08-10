#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Test2D 
{
public:

  Test2D(size_t a_size){}  

  virtual void Run(const int a_size, int* outData1ui [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel1D_MemSet(const int a_size, int a_offset, int a_val, int* outData1ui);
  void kernel2D_MemSet(const int a_width, int a_height, int a_pitch, int a_val, int* outData1ui);
};

#endif