#pragma once
#include <vector>
#include <cstdint>

#include "neural.h"
#include "LiteMath.h"
using LiteMath::half;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::uint4;

// the same networks as in test 072, but layers are addressed via table 'm_layers': on GPU weights are repacked to 'InferencingOptimal' layout
static constexpr uint32_t NN_INPUT_DIM  = 6;
static constexpr uint32_t NN_HIDDEN_DIM = 32;
static constexpr uint32_t NN_OUTPUT_DIM = 16;
static constexpr uint32_t NN_LAYERS     = 3;

class MultiNet
{
public:
  MultiNet(){}

  void AddNetwork(const float* a_matrices[NN_LAYERS], const float* a_biases[NN_LAYERS])
  {
    const uint32_t dims[NN_LAYERS+1] = {NN_INPUT_DIM, NN_HIDDEN_DIM, NN_HIDDEN_DIM, NN_OUTPUT_DIM};
    for(uint32_t layer = 0; layer < NN_LAYERS; layer++)
      m_layers.push_back(LiteNN::AppendLayer(m_weights, a_matrices[layer], a_biases[layer], dims[layer], dims[layer+1]));
  }

  virtual void Run(const uint32_t* a_netId [[size("a_size")]], const float* a_in [[size("a_size*6")]], const uint4* a_idx [[size("a_size")]],
                   float4* a_out [[size("a_size")]], uint32_t a_size);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  void kernel1D_Eval(const uint32_t* a_netId, const float* a_in, const uint4* a_idx, float4* a_out, uint32_t a_size);

  float4 EvalNetwork(uint32_t a_netId, float3 a_pos, float3 a_dir, uint4 a_idx) const;

  std::vector<half>          m_weights;
  std::vector<LiteNN::Layer> m_layers; // [netId*NN_LAYERS + layer]
};
