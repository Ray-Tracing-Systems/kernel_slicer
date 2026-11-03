#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <cstdint>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Pointer to a member vector is passed to a function and then further through several levels of nested functions,
// the position inside the vector is given by a separate offset argument and by 'pointer + k' at a call site.
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
