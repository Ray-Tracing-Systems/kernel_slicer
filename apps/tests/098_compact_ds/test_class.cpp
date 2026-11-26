#include "test_class.h"
#include <cstring>

static inline float SumTwo(const float* a_data, uint32_t a_id)
{
  return a_data[a_id] + a_data[a_id + 1];
}

static inline uint32_t PickId(const uint32_t* a_ids, uint32_t a_offset, uint32_t a_id)
{
  return a_ids[a_offset + a_id];
}

void TestCompactDS::InitData(int a_iter)
{
  // 'm_fA', 'm_uA', 'm_uB' are only appended after the first iteration (they are in append group 'geom' of kmake_append.json),
  // other vectors are filled again with other sizes and values
  //
  const size_t sizeFA = 37 + 50*a_iter;
  const size_t sizeUA = 29 + 3*a_iter;
  const size_t sizeUB = 7  + 100*a_iter;
  for(size_t i=m_fA.size(); i<sizeFA; i++) m_fA.push_back(float(i % 11) + 0.25f);
  for(size_t i=m_uA.size(); i<sizeUA; i++) m_uA.push_back(uint32_t((i*13) % 97));
  for(size_t i=m_uB.size(); i<sizeUB; i++) m_uB.push_back(uint32_t((i*5)  % 31));

  const float scale = 1.0f + 0.5f*float(a_iter);
  m_fB.resize(100 + 17*a_iter);
  m_fC.resize(13 + a_iter);
  m_fEmpty.clear();
  m_colA.resize(11 + a_iter);
  m_colB.resize(5);
  m_iA.resize(9 + 2*a_iter);
  m_materials.resize(3 + a_iter);
  m_written.resize(64);
  m_copy.resize(m_fC.size());
  m_texA.resize(8, 1);
  m_sampler.filter = Sampler::Filter::NEAREST;

  for(size_t i=0;i<m_fB.size();i++)      m_fB[i]   = scale*float(i % 7);
  for(size_t i=0;i<m_fC.size();i++)      m_fC[i]   = scale*float(i % 5) + 0.5f;
  for(size_t i=0;i<m_colA.size();i++)    m_colA[i] = float4(float(i), scale, 0.0f, 1.0f);
  for(size_t i=0;i<m_colB.size();i++)    m_colB[i] = float4(0.0f, scale*float(i % 3), 2.0f, 0.0f);
  for(size_t i=0;i<m_iA.size();i++)      m_iA[i]   = int(i % 4) - 2;
  for(size_t i=0;i<m_materials.size();i++)
  {
    m_materials[i].color = float4(scale, 0.5f, 0.25f, 1.0f);
    m_materials[i].rough = 0.1f*float(i + 1);
    m_materials[i].texId = uint32_t(i*3 + a_iter);
    m_materials[i].pad   = float2(0.0f);
  }
  for(int x=0;x<8;x++)                      // the same for all iterations: CommitAppendedData() does not copy textures
    m_texA[LiteMath::int2(x, 0)] = float4(0.25f*float(x) + 1.0f, 0.0f, 0.0f, 1.0f);
}

float TestCompactDS::GetColor(uint32_t a_id) const
{
  const uint32_t matId = a_id % uint32_t(m_materials.size());
  return m_colA[a_id % m_colA.size()].x + m_colB[a_id % m_colB.size()].y*m_materials[matId].rough + float(m_materials[matId].texId);
}

void TestCompactDS::kernel1D_Write(uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++)
    m_written[i] = 3.0f*m_fA[i % m_fA.size()] + 1.0f;
}

void TestCompactDS::kernel1D_Main(uint32_t a_size, float* a_out)
{
  for(uint32_t i=0; i<a_size; i++)
  {
    const uint32_t j = m_uA[i % m_uA.size()];
    const uint32_t k = PickId(m_uB.data(), 1, i % uint32_t(m_uB.size() - 1));
    float v = m_fA[j % m_fA.size()] + 2.0f*m_fB[k % m_fB.size()];
    v += SumTwo(m_fC.data(), i % 5);
    v += 0.5f*SumTwo(m_fC.data() + 3, i % 3);
    const float* p = m_fB.data() + 1;
    v += p[i % 4];
    v += GetColor(i);
    v += float(m_iA[i % m_iA.size()]);
    v += 10.0f*float(m_fEmpty.size());
    for(uint32_t t=0; t<m_fEmpty.size(); t++)
      v += m_fEmpty[t];
    v += m_written[i % m_written.size()];
    v += m_copy[i % m_copy.size()];
    v += m_texA.sample(m_sampler, float2((float(i % 8) + 0.5f)/8.0f, 0.5f)).x;
    a_out[i] = v;
  }
}

void TestCompactDS::kernel1D_FromArg(uint32_t a_size, const float* a_in, uint32_t a_inSize, float* a_out)
{
  for(uint32_t i=0; i<a_size; i++)
    a_out[i] += 100.0f*a_in[i % a_inSize];
}

void TestCompactDS::Run(uint32_t a_size, float* a_out)
{
  kernel1D_Write(uint32_t(m_written.size()));
  memcpy(m_copy.data(), m_fC.data(), m_fC.size()*sizeof(float));
  kernel1D_Main(a_size, a_out);
  kernel1D_FromArg(a_size, m_fB.data(), uint32_t(m_fB.size()), a_out);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void TestCompactDS::RunRTVBlock(uint32_t a_size, float* a_out, uint32_t a_passesNum)
{
  for(uint32_t tid=0; tid<a_size; tid++)
    RunRTV(tid, a_out);
}

void TestCompactDS::RunRTV(uint32_t tid, float* a_out)
{
  float val;
  kernel_Eval (tid, &val);
  kernel_Store(tid, &val, a_out);
}

void TestCompactDS::kernel_Eval(uint32_t tid, float* a_val)
{
  *a_val = m_fA[tid % m_fA.size()] + float(m_uB[tid % m_uB.size()]) + GetColor(tid) + SumTwo(m_fB.data() + 2, tid % 7);
}

void TestCompactDS::kernel_Store(uint32_t tid, const float* a_val, float* a_out)
{
  a_out[tid] = (*a_val)*2.0f + m_fC[tid % m_fC.size()] + m_colB[tid % m_colB.size()].z;
}
