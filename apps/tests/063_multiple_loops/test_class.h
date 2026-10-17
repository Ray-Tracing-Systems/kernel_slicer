#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>

// Several parallel loops inside one kernel; single-threaded sections between them may declare variables
class MultiLoops
{
public:
  MultiLoops(){}

  void SetMaxSize(uint32_t a_size) { m_tmp.resize(a_size); }

  virtual void Run(const float* a_in [[size("a_size")]], float* a_out [[size("a_size")]], uint32_t a_size);
  virtual void Blur(const float* a_in [[size("a_width*a_height")]], float* a_out [[size("a_width*a_height")]], uint32_t a_width, uint32_t a_height);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_TwoPasses(const float* a_in, float* a_out, uint32_t a_size);
  void kernel2D_Blur(const float* a_in, float* a_out, uint32_t a_width, uint32_t a_height);

  std::vector<float> m_tmp;
  uint32_t           m_count  = 0;
  float              m_summ   = 0.0f;
  float              m_result = 0.0f;
};
