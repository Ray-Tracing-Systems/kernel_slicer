#include "test_class.h"

void PtrCast::Run(const float* a_in, float* a_out, uint32_t a_size)
{
  // second half of a_out (4*a_size floats from offset 4*a_size) is used as 'a_size' cells
  kernel1D_MakeCells(a_in, (Cell*)a_out, a_size, a_size);
  // member vector of floats is used as cells: a copy of the cells with scaled weights
  kernel1D_ScaleCells(reinterpret_cast<Cell*>(m_scratch.data()), 0, (const Cell*)a_out, a_size, a_size);
  // and back to the cells of a_out, scaled again
  kernel1D_ScaleCells(static_cast<Cell*>((Cell*)a_out), a_size, reinterpret_cast<const Cell*>(m_scratch.data()), 0, a_size);
  // one buffer is passed twice: cells of the second half are read, the first half gets the result (4 floats per element)
  kernel1D_Compose((const Cell*)a_out, a_size, a_out, a_size);
}

void PtrCast::kernel1D_MakeCells(const float* a_in, Cell* a_cells, uint32_t a_cellOffset, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
  {
    Cell c;
    c.value  = a_in[i] * 2.0f;
    c.index  = i * 3u + 1u;
    c.weight = 0.5f + float(i % 7);
    c.pad    = -1.0f;
    a_cells[a_cellOffset + i] = c;
  }
}

void PtrCast::kernel1D_ScaleCells(Cell* a_cells, uint32_t a_cellOffset, const Cell* a_src, uint32_t a_srcOffset, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
  {
    Cell c = a_src[a_srcOffset + i];
    c.weight *= 0.25f;
    a_cells[a_cellOffset + i] = c;
  }
}

void PtrCast::kernel1D_Compose(const Cell* a_cells, uint32_t a_cellOffset, float* a_color, uint32_t a_size)
{
  for(uint32_t i = 0; i < a_size; i++)
  {
    const Cell c = a_cells[a_cellOffset + i];
    a_color[i*4 + 0] = c.value * c.weight;
    a_color[i*4 + 1] = float(c.index);
    a_color[i*4 + 2] = c.weight;
    a_color[i*4 + 3] = c.pad;
  }
}
