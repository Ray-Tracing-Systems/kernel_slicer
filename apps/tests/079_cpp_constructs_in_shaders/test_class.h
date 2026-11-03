#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <cstdint>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Ordinary C++ which shader languages do not have directly: INFINITY, std::nextafter, C-style cast in a global constant,
// local 'static constexpr', default arguments, implicit conversions to bool and bool members of the class (4 bytes in UBO on GPU).
//
class TestCppConstructs 
{
public:

  TestCppConstructs(size_t a_size);  

  virtual void Run(const int a_size, float* outData [[size("a_size")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel1D_Run(const int a_size, float* outData);
  std::vector<float> m_data;
  bool m_flagA = true;
  bool m_flagB = false;
  bool m_flagC = true;
};

#endif
