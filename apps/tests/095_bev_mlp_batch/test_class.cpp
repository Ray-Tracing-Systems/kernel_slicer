#include "test_class.h"
#include <algorithm>

void BatchMLP::Run(const float* a_in, float* a_out, uint a_size)
{
  kernelBE1D_Eval<THREADS>((a_size + BATCH - 1)/BATCH, a_in, a_out, a_size);
}

// One block evaluates the network for BATCH vectors: layers are block level functions (cooperative matrices on GPU),
// activations between them are computed by parallel loops.
//
template<uint bsize>
void BatchMLP::kernelBE1D_Eval(uint a_blockNum, const float* a_in, float* a_out, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    half  act[BATCH*NN_HIDDEN_DIM];                       // inputs of a layer for the batch, fp16 (shared memory)
    float res[BATCH*NN_HIDDEN_DIM];                       // outputs of a layer for the batch, fp32 (shared memory)

    for(uint i = 0; i < BATCH*NN_INPUT_DIM; i++) [[parallel]]   // load inputs, zero vectors after the end of the array
    {
      const uint vec = blockId*BATCH + i / NN_INPUT_DIM;
      act[i] = half((vec < a_size) ? a_in[vec*NN_INPUT_DIM + i % NN_INPUT_DIM] : 0.0f);
    }

    LiteNN::LinearForwardBlock(m_weights.data(), m_matOffs[0], m_biasOffs[0], act, NN_INPUT_DIM, res, NN_HIDDEN_DIM, BATCH);
    for(uint i = 0; i < BATCH*NN_HIDDEN_DIM; i++) [[parallel]]
      act[i] = half(std::max(res[i], 0.0f));

    LiteNN::LinearForwardBlock(m_weights.data(), m_matOffs[1], m_biasOffs[1], act, NN_HIDDEN_DIM, res, NN_HIDDEN_DIM, BATCH);
    for(uint i = 0; i < BATCH*NN_HIDDEN_DIM; i++) [[parallel]]
      act[i] = half(std::max(res[i], 0.0f));

    LiteNN::LinearForwardBlock(m_weights.data(), m_matOffs[2], m_biasOffs[2], act, NN_HIDDEN_DIM, res, NN_OUTPUT_DIM, BATCH);

    for(uint i = 0; i < BATCH*NN_OUTPUT_DIM; i++) [[parallel]]  // store outputs of existing vectors
    {
      const uint vec = blockId*BATCH + i / NN_OUTPUT_DIM;
      if(vec < a_size)
        a_out[vec*NN_OUTPUT_DIM + i % NN_OUTPUT_DIM] = res[i];
    }
  }
}
