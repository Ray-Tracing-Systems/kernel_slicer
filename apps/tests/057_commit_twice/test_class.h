#pragma once
#include <vector>
#include <cstdint>

// Checks that a repeated CommitDeviceData() call frees and re-creates all device data correctly:
// member buffers of a different size, reduction buffers, indirect dispatch, memcpy service, RTV local (per-thread) buffers.
class CommitTest
{
public:
  CommitTest(){}

  void SetData(const std::vector<float>& a_data);

  virtual void Process(float* a_out [[size("a_size")]], unsigned int a_size);

  virtual void ProcessRTVBlock(float* a_out, unsigned int a_size, unsigned int a_passesNum);
  void ProcessRTV(float* a_out [[size("tid")]], unsigned int tid);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Summ(const float* a_data);                                    // reduction to m_summ, indirect (loop bound is a class member)
  void kernel1D_Scale(float* a_out, unsigned int a_size);

  void kernel_Load (float* a_val, unsigned int tid);
  void kernel_Store(float* a_out, const float* a_val, unsigned int tid);

  float              m_summ  = 0.0f;
  unsigned int       m_count = 0;
  std::vector<float> m_data;
  std::vector<float> m_copy;
};
