#pragma once
#include <vector>
#include <cstdint>

// 2D kernels with loop bounds from class members (indirect dispatch), translated with '-reorderLoops YX'
class IndirectReorder
{
public:
  IndirectReorder(){}

  void SetSize(uint32_t a_width, uint32_t a_height) { m_width = a_width; m_height = a_height; m_image.resize(a_width*a_height); }

  virtual void Run(uint32_t* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel2D_Fill();                     // indirect, both bounds are members
  void kernel2D_Copy(uint32_t* a_out);      // indirect, both bounds are members
  void kernel1D_Shrink(uint32_t a_threads);                 // changes m_height on GPU, so the next dispatch size is known only on GPU

  uint32_t m_width  = 0;
  uint32_t m_height = 0;
  std::vector<uint32_t> m_image;
};
