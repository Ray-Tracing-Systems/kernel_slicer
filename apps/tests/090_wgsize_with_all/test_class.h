#pragma once
#include <vector>
#include <cstdint>

// kmake.json has both "all" (nonConstantData for every kernel) and a named kernel entry with "wgSize":
// "all" must override only the keys it has, the work group size of the named kernel must be kept
class WgSizeAll
{
public:
  WgSizeAll(){}

  virtual void Run(const float* a_in [[size("a_size")]], float* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void UpdateMembersPlainData() {}                                 // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

protected:
  void kernel1D_Scale(const float* a_in, float* a_out, uint32_t a_size);
  void kernel2D_Fill(uint32_t a_width, uint32_t a_height, float* a_out);
};
