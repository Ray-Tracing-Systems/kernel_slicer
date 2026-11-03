#include "test_class.h"
void SimpleMLP::kernel1D_Eval(const float* a_in, float* a_out, uint32_t a_size)
{
  for(uint32_t i=0; i<a_size; i++)
  {
    float buf0[NN_HIDDEN_DIM];
    float buf1[NN_HIDDEN_DIM];
    buf0[0] = a_in[i*3+0];
    buf0[1] = a_in[i*3+1];
    buf0[2] = a_in[i*3+2];

    LiteNN::LinearForward_ReLU(m_weights.data(), m_matOffs[0], m_biasOffs[0], buf0, NN_INPUT_DIM,  buf1, NN_HIDDEN_DIM);
    LiteNN::LinearForward(m_weights.data(), m_matOffs[1], m_biasOffs[1], buf1, NN_HIDDEN_DIM, buf0, NN_HIDDEN_DIM);
    LiteNN::ReLU(buf0, NN_HIDDEN_DIM);
    LiteNN::LinearForward(m_weights.data(), m_matOffs[2], m_biasOffs[2], buf0, NN_HIDDEN_DIM, buf1, NN_OUTPUT_DIM);

    for(uint32_t j=0; j<NN_OUTPUT_DIM; j++)
      a_out[i*NN_OUTPUT_DIM+j] = buf1[j];
  }
}

void SimpleMLP::Run(const float* a_in, float* a_out, uint32_t a_size)
{
  kernel1D_Eval(a_in, a_out, a_size);
}
