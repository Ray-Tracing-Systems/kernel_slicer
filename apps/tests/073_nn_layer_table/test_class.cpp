#include "test_class.h"

float4 MultiNet::EvalNetwork(uint32_t a_netId, float3 a_pos, float3 a_dir, uint4 a_idx) const
{
  const uint32_t first = a_netId*NN_LAYERS;

  float buf0[NN_HIDDEN_DIM];
  float buf1[NN_HIDDEN_DIM];
  buf0[0] = a_pos.x;
  buf0[1] = a_pos.y;
  buf0[2] = a_pos.z;
  buf0[3] = a_dir.x;
  buf0[4] = a_dir.y;
  buf0[5] = a_dir.z;

  LiteNN::LinearForward_ReLU(m_weights.data(), m_layers[first + 0], buf0, NN_INPUT_DIM,  buf1, NN_HIDDEN_DIM);
  LiteNN::LinearForward_ReLU(m_weights.data(), m_layers[first + 1], buf1, NN_HIDDEN_DIM, buf0, NN_HIDDEN_DIM);
  LiteNN::LinearForward     (m_weights.data(), m_layers[first + 2], buf0, NN_HIDDEN_DIM, buf1, NN_OUTPUT_DIM);

  return float4(buf1[a_idx.x], buf1[a_idx.y], buf1[a_idx.z], buf1[a_idx.w]);
}

void MultiNet::kernel1D_Eval(const uint32_t* a_netId, const float* a_in, const uint4* a_idx, float4* a_out, uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++)
  {
    const float3 pos = float3(a_in[i*6+0], a_in[i*6+1], a_in[i*6+2]);
    const float3 dir = float3(a_in[i*6+3], a_in[i*6+4], a_in[i*6+5]);
    a_out[i] = EvalNetwork(a_netId[i], pos, dir, a_idx[i]);
  }
}

void MultiNet::Run(const uint32_t* a_netId, const float* a_in, const uint4* a_idx, float4* a_out, uint32_t a_size)
{
  kernel1D_Eval(a_netId, a_in, a_idx, a_out, a_size);
}
