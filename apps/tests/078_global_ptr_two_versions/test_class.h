#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <cstdint>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The same functions are called with pointer to local variable and with pointer to global memory (member vector).
// Both versions are needed on GPU: original function for local variable and its copy for buffer;
// the original version calls original versions of other functions, the copy calls their copies.
//
class TestGlobalPtrTwoVersions
{
public:

  TestGlobalPtrTwoVersions(size_t a_size);

  virtual void Run(const int a_size, float* outData [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:

  void kernel1D_Run(const int a_size, float* outData);
  std::vector<float> m_params;
};

#endif
