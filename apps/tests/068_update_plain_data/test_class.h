#pragma once
#include <vector>
#include <cstdint>

// Host changes POD members and calls UpdateMembersPlainData() instead of CommitDeviceData():
// only plain members (UBO) are copied to GPU again, member vectors keep their contents
class PlainUpdate
{
public:
  PlainUpdate(){}

  void SetTable(const std::vector<float>& a_table) { m_table = a_table; }
  void SetParameters(float a_scale, float a_offset, uint32_t a_shift) { m_scale = a_scale; m_offset = a_offset; m_shift = a_shift; }

  virtual void Run(const float* a_in [[size("a_size")]], float* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void UpdateMembersPlainData() {}                                 // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  void kernel1D_Transform(const float* a_in, float* a_out, uint32_t a_size);

  std::vector<float> m_table;
  float              m_scale  = 1.0f;
  float              m_offset = 0.0f;
  uint32_t           m_shift  = 0;
};
