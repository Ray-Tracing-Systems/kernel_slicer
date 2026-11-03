#include "test_class.h"

static inline Accum AddValue(Accum a_acc, float a_val)
{
  a_acc.count++;
  a_acc.sum += a_val;
  if(a_val > a_acc.maxV)
    a_acc.maxV = a_val;
  return a_acc;
}

/////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////

TestStructDefaultInit::TestStructDefaultInit(size_t a_size)
{
  m_data.resize(a_size);
  for(size_t i=0;i<m_data.size();i++)
    m_data[i] = float(i % 13);
}

void TestStructDefaultInit::kernel1D_Run(const int a_size, float* outData)
{
  for(int i=0;i<a_size;i++)
  {
    Accum acc;            // all fields are initialized explicitly: default member initializers are removed in GLSL
    acc.count = 0;
    acc.sum   = 0.0f;
    acc.maxV  = -1.0f;
    acc.flags = 5;
    acc.isBig = 1;

    Pair pair;
    pair.weights[0] = 0.5f;
    pair.weights[1] = 0.25f;
    pair.id         = 7;

    for(int j=0;j<4;j++)
      acc = AddValue(acc, m_data[(i + j) % a_size]);

    outData[i] = acc.sum*pair.weights[0] + acc.maxV*pair.weights[1] + float(acc.count + acc.flags + acc.isBig + pair.id);
  }
}

void TestStructDefaultInit::Run(const int a_size, float* outData)
{
  kernel1D_Run(a_size, outData);
}
