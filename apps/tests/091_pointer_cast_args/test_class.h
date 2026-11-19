#pragma once
#include <vector>
#include <cstdint>

/// 16 bytes, the same size as 4 floats: a part of a float buffer is used as an array of cells
struct Cell
{
  float    value;
  uint32_t index;
  float    weight;
  float    pad;
};

/// Pointer casts in arguments of kernel calls: '(Cell*)a_out', 'reinterpret_cast<Cell*>(m_scratch.data())', nested 'static_cast<Cell*>((Cell*)a_out)'.
/// One buffer is used as a storage of different types (offsets are passed as separate arguments, in elements of the kernel type),
/// in kernel1D_Compose it is passed twice: as 'const Cell*' (read) and as 'float*' (write), the ranges do not overlap.
class PtrCast
{
public:
  PtrCast(){}

  virtual void Run(const float* a_in [[size("a_size")]], float* a_out [[size("a_size*8")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void UpdateMembersPlainData() {}                                 // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  std::vector<float> m_scratch; ///< 4 floats per cell, resized by host

protected:
  void kernel1D_MakeCells (const float* a_in, Cell* a_cells, uint32_t a_cellOffset, uint32_t a_size);
  void kernel1D_ScaleCells(Cell* a_cells, uint32_t a_cellOffset, const Cell* a_src, uint32_t a_srcOffset, uint32_t a_size);
  void kernel1D_Compose   (const Cell* a_cells, uint32_t a_cellOffset, float* a_color, uint32_t a_size);
};
