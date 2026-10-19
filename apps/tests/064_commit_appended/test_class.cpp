#include "test_class.h"

void AppendTest::kernel1D_DumpA(uint32_t* a_out)
{
  for(uint32_t i = 0; i < m_a.size(); i++)
    a_out[i] = m_a[i]*3u + 1u;
}

void AppendTest::kernel1D_DumpB(uint2* a_out)
{
  for(uint32_t i = 0; i < m_b.size(); i++)
  {
    const uint2 val = m_b[i];
    a_out[i] = uint2(val.x + 7u, val.y*5u);
  }
}

void AppendTest::kernel1D_DumpC(uint32_t* a_out)
{
  for(uint32_t i = 0; i < m_c.size(); i++)
    a_out[i] = m_c[i] ^ 0x5A5A5A5Au;
}

void AppendTest::kernel1D_DumpSizes(uint32_t a_one, uint32_t* a_out)
{
  for(uint32_t i = 0; i < a_one; i++)
  {
    a_out[0] = uint32_t(m_a.size());
    a_out[1] = uint32_t(m_b.size());
    a_out[2] = uint32_t(m_c.size());
    a_out[3] = 12345u;
  }
}

void AppendTest::Dump(uint32_t a_maxSize, uint32_t* a_outA, uint2* a_outB, uint32_t* a_outC, uint32_t* a_outSizes)
{
  kernel1D_DumpA(a_outA);
  kernel1D_DumpB(a_outB);
  kernel1D_DumpC(a_outC);
  kernel1D_DumpSizes(1, a_outSizes);
}
