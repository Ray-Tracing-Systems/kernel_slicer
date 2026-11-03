#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <cstdint>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The same as 074_nested_global_ptr, but the pointer is shifted ('pointer + k') in kernel code and when it is passed from one nested function to another.
//
class TestNestedGlobalPtr
{
public:

  TestNestedGlobalPtr(size_t a_size);

  virtual void Run(const int a_size, float* outData [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:

  void kernel1D_Run(const int a_size, float* outData);
  std::vector<float> m_params;
};

#endif
