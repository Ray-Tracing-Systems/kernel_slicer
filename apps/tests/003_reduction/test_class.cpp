#include "test_class.h"
#include <cassert>
#include <algorithm>
#include <memory>
#include <cstring>

using LiteMath::min;
using LiteMath::max;

void BoxMinMax::kernel1D_FindBoundingBox(const float4* a_inData, uint32_t a_dataSize, float4* a_outData)
{
  m_boxMin = float4(+1e6f,+1e6f,+1e6f,+1e6f);
  m_boxMax = float4(-1e6f,-1e6f,-1e6f,-1e6f);
  
  for(uint32_t i=0; i<a_dataSize; i++)
  {
    float4 p = a_inData[i];
    m_boxMin = min(m_boxMin, p);
    m_boxMax = max(m_boxMax, p);
  }

  a_outData[0] = m_boxMin;
  a_outData[1] = m_boxMax;
}

void BoxMinMax::ProcessPoints(const float4* a_inData, size_t a_dataSize, float4* a_outData)
{
  kernel1D_FindBoundingBox(a_inData, uint32_t(a_dataSize), a_outData);
}
