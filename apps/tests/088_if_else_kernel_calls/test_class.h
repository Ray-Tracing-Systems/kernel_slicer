#pragma once
#include <vector>
#include <cstdint>

// Kernel calls in 'if / else if / else' of a control function without braces:
// the generated command text must stay one statement, otherwise 'else' is a syntax error
class IfElseCalls
{
public:
  IfElseCalls(){}

  void SetMode(uint32_t a_mode) { m_mode = a_mode; }

  virtual void Run(const float* a_in [[size("a_size")]], float* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void UpdateMembersPlainData() {}                                 // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  void kernel1D_Add(const float* a_in, float* a_out, uint32_t a_size);
  void kernel1D_Mul(const float* a_in, float* a_out, uint32_t a_size);
  void kernel1D_Neg(float* a_data, uint32_t a_size);

  uint32_t m_mode = 0;
};
