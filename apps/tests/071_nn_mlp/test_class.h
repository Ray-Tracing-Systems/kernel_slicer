#pragma once
#include <vector>
#include <cstdint>

#include "neural.h"
using LiteMath::half;

// MLP 3 -> 16 -> 16 -> 4 with ReLU, weights in fp16 (layout 'A': offsets of layers are plain class members)
static constexpr uint32_t NN_INPUT_DIM  = 3;
static constexpr uint32_t NN_HIDDEN_DIM = 16;
static constexpr uint32_t NN_OUTPUT_DIM = 4;

class SimpleMLP
{
public:
  SimpleMLP(){}

  void SetLayer(int a_layer, const float* a_matrix, const float* a_bias, uint32_t a_inDim, uint32_t a_outDim)
  {
    if(a_layer == 0)
      m_weights.clear();
    auto offs = LiteNN::AppendLayer(m_weights, a_matrix, a_bias, a_inDim, a_outDim);
    m_matOffs[a_layer]  = offs.matOffs;
    m_biasOffs[a_layer] = offs.biasOffs;
  }

  virtual void Run(const float* a_in [[size("a_size*3")]], float* a_out [[size("a_size*4")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Eval(const float* a_in, float* a_out, uint32_t a_size);

  std::vector<half> m_weights;
  uint32_t m_matOffs[3];
  uint32_t m_biasOffs[3];
};
