#include "test_class.h"

void HashMapAlign::Reserve(uint32_t a_maxKeys)
{
  m_f2.reserve(a_maxKeys);
  m_f3.reserve(a_maxKeys);
  m_f4.reserve(a_maxKeys);
  m_part.reserve(a_maxKeys);
  m_sum.reserve(a_maxKeys);
  m_cnt.reserve(a_maxKeys);
  m_neg.reserve(a_maxKeys);
}

void HashMapAlign::Build(const uint32_t* a_keys, uint32_t a_keysNum, const uint32_t* a_dupKeys, uint32_t a_dupKeysNum)
{
  kernel1D_Fill(a_keys, a_keysNum);
  kernel1D_Accum(a_dupKeys, a_dupKeysNum);
}

void HashMapAlign::Lookup(const uint32_t* a_queries, uint32_t a_queryNum, float4* a_out)
{
  kernel1D_Lookup(a_queries, a_queryNum, a_out);
}

void HashMapAlign::kernel1D_Fill(const uint32_t* a_keys, uint32_t a_keysNum)
{
  for(uint32_t i=0; i < a_keysNum; i++)
  {
    const uint32_t key = a_keys[i];
    m_f2[key]   = ValueF2(key);
    m_f3[key]   = ValueF3(key);
    m_f4[key]   = ValueF4(key);
    m_part[key] = ValuePart(key);
  }
}

void HashMapAlign::kernel1D_Accum(const uint32_t* a_dupKeys, uint32_t a_dupKeysNum)
{
  for(uint32_t i=0; i < a_dupKeysNum; i++)
  {
    const uint32_t key = a_dupKeys[i];
    m_sum[key] += 0.5f;
    m_cnt[key]++;
    m_neg[key] -= 2;
    m_part[key].vel.x += 1.0f; // atomic update of struct field: all 'a_dupKeys' are present in 'm_part' after kernel1D_Fill
  }
}

float HashMapAlign::GetSum(uint32_t a_key) const
{
  const auto it = m_sum.find(a_key);
  return (it != m_sum.end()) ? it->second : -1.0f;
}

void HashMapAlign::kernel1D_Lookup(const uint32_t* a_queries, uint32_t a_queryNum, float4* a_out)
{
  for(uint32_t i=0; i < a_queryNum; i++)
  {
    const uint32_t key = a_queries[i];

    float4 res0 = float4(-1.0f);
    float4 res1 = float4(-1.0f);
    float4 res2 = float4(-1.0f);
    float4 res3 = float4(-1.0f);

    if(m_f2.count(key) != 0)
    {
      const float2 f2 = m_f2.at(key);
      const float3 f3 = m_f3.at(key);
      res0 = float4(f2.x, f2.y, f3.y, f3.z);
      res1 = m_f4.at(key);
      res2 = m_part.at(key).vel;
    }

    const auto pCnt = m_cnt.find(key);
    if(pCnt != m_cnt.end())
    {
      res3.x = float(pCnt->second);
      res3.y = float(pCnt->first % 1000);
      res3.z = float(m_neg.at(key));
    }
    res3.w = GetSum(key);

    if(i == 0)
      res3.w = float(m_cnt.size()); // number of unique keys in 'a_dupKeys'

    a_out[i*4+0] = res0;
    a_out[i*4+1] = res1;
    a_out[i*4+2] = res2;
    a_out[i*4+3] = res3;
  }
}
