#pragma once
#include <vector>
#include <cstdint>

// 16-bit fields inside a structure stored in a member vector; kslicer must request 'storageBuffer16BitAccess' automatically
struct PackedVertex
{
  uint16_t u;
  uint16_t v;
  uint32_t id;
  uint16_t flags[2];
};

class Storage16
{
public:
  Storage16(){}

  void SetData(const std::vector<PackedVertex>& a_data) { m_verts = a_data; m_bias = a_data[1]; }

  virtual void Run(float* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Unpack(float* a_out, uint32_t a_size);

  std::vector<PackedVertex> m_verts;
  PackedVertex              m_bias;     // plain member with 16-bit fields, placed in UBO
};
