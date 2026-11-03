#include "test_class.h"

static inline float c_read(const float* a_params, uint32_t a_id)
{
  return a_params[a_id];
}

static inline float b_sum(const float* a_params, uint32_t a_id)
{
  return c_read(a_params, a_id) + 10.0f*c_read(a_params + 1, a_id);
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

TestLocalGlobalPtrVar::TestLocalGlobalPtrVar(size_t a_size)
{
  m_params.resize(64);
  for(size_t i=0;i<m_params.size();i++)
    m_params[i] = float(i % 10);

  m_pairs.resize(5);
  for(size_t i=0;i<m_pairs.size();i++)
  {
    m_pairs[i].x = float(i);
    m_pairs[i].y = float(2*i + 1);
  }
}

float TestLocalGlobalPtrVar::evalNode(uint32_t a_offset, uint32_t a_id)
{
  const float* points = &m_params[a_offset + 4];                 // the same as 'node parameters' in a tree stored in flat array
  return points[a_id] + 10.0f*b_sum(points, a_id);
}

void TestLocalGlobalPtrVar::kernel1D_Run(const int a_size, float* outData)
{
  for(int i=0;i<a_size;i++)
  {
    const uint32_t offset = uint32_t(i % 4)*8;
    const uint32_t id     = uint32_t(i % 3);

    const float* p  = &m_params[offset];                          // '&m_vec[e]'
    const float* q  = m_params.data() + offset + 2;               // 'm_vec.data() + e'
    const float* r  = p + 1;                                      // 'p + e'
    const Pair*  pp = &m_pairs[uint32_t(i) % 5];

    float res = p[1] + 2.0f*(*q) + 3.0f*r[id];                    // 'p[i]', '*p'
    res += 100.0f*b_sum(p, id) + 1000.0f*b_sum(q + 1, id) + 10000.0f*b_sum(r, 0); // passing 'p' and 'p + k' to functions
    res += pp->x + pp->y;                                         // 'p->x'
    res += 10000.0f*evalNode(offset, id);                         // all terms are integers, sum < 2^24: exact on CPU and GPU

    outData[i] = res;
  }
}

void TestLocalGlobalPtrVar::Run(const int a_size, float* outData)
{
  kernel1D_Run(a_size, outData);
}
