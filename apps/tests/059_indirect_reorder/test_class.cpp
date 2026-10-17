#include "test_class.h"

void IndirectReorder::kernel2D_Fill()
{
  for(uint32_t y=0; y<m_height; y++)
    for(uint32_t x=0; x<m_width; x++)
      m_image[y*m_width + x] = x + y*1000 + 1;
}

void IndirectReorder::kernel1D_Shrink(uint32_t a_threads)
{
  for(uint32_t i=0; i<a_threads; i++)
    m_height = m_height - 3;
}

void IndirectReorder::kernel2D_Copy(uint32_t* a_out)
{
  for(uint32_t y=0; y<m_height; y++)
    for(uint32_t x=0; x<m_width; x++)
      a_out[y*m_width + x] = m_image[y*m_width + x]*2;
}

void IndirectReorder::Run(uint32_t* a_out, uint32_t a_size)
{
  kernel2D_Fill();
  kernel1D_Shrink(1);
  kernel2D_Copy(a_out);
}
