#include "test_class.h"
#include <cstring>

void CommitTest::SetData(const std::vector<float>& a_data)
{
  m_data  = a_data;
  m_count = (unsigned int)(a_data.size());
  m_copy.resize(a_data.size());
}

void CommitTest::kernel1D_Summ(const float* a_data)
{
  m_summ = 0.0f;
  for(unsigned int i=0; i<m_count; i++)
    m_summ += a_data[i];
}

void CommitTest::kernel1D_Scale(float* a_out, unsigned int a_size)
{
  for(unsigned int i=0; i<a_size; i++)
    a_out[i] = m_copy[i % m_count]*m_summ;
}

void CommitTest::Process(float* a_out, unsigned int a_size)
{
  kernel1D_Summ(m_data.data());
  memcpy(m_copy.data(), m_data.data(), m_data.size()*sizeof(float));
  kernel1D_Scale(a_out, a_size);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void CommitTest::ProcessRTVBlock(float* a_out, unsigned int a_size, unsigned int a_passesNum)
{
  for(unsigned int tid=0; tid<a_size; tid++)
    ProcessRTV(a_out, tid);
}

void CommitTest::ProcessRTV(float* a_out, unsigned int tid)
{
  float val;
  kernel_Load(&val, tid);
  kernel_Store(a_out, &val, tid);
}

void CommitTest::kernel_Load(float* a_val, unsigned int tid)
{
  *a_val = m_data[tid % m_count] + float(tid);
}

void CommitTest::kernel_Store(float* a_out, const float* a_val, unsigned int tid)
{
  a_out[tid] = (*a_val)*2.0f;
}
