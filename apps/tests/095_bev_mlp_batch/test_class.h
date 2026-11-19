#pragma once
#include <vector>
#include <cstdint>

#include "neural.h"
using LiteMath::half;
using LiteMath::uint;

// MLP 16 -> 64 -> 64 -> 16 with ReLU, weights in fp16; one block evaluates a batch of BATCH vectors
static constexpr uint NN_INPUT_DIM  = 16;
static constexpr uint NN_HIDDEN_DIM = 64;
static constexpr uint NN_OUTPUT_DIM = 16;
static constexpr uint BATCH         = 64;    // vectors per block
static constexpr uint THREADS       = 128;   // threads per block

class BatchMLP
{
public:
  BatchMLP(){}

  void SetLayer(int a_layer, const float* a_matrix, const float* a_bias, uint32_t a_inDim, uint32_t a_outDim)
  {
    if(a_layer == 0)
      m_weights.clear();
    auto offs = LiteNN::AppendLayer(m_weights, a_matrix, a_bias, a_inDim, a_outDim);
    m_matOffs[a_layer]  = offs.matOffs;
    m_biasOffs[a_layer] = offs.biasOffs;
  }

  virtual void Run(const float* a_in [[size("a_size*NN_INPUT_DIM")]], float* a_out [[size("a_size*NN_OUTPUT_DIM")]], uint a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  template<uint bsize>
  void kernelBE1D_Eval(uint a_blockNum, const float* a_in, float* a_out, uint a_size);

  std::vector<half> m_weights;
  uint32_t m_matOffs[3];
  uint32_t m_biasOffs[3];
};
