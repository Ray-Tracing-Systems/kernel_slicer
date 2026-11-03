#pragma once
////////////////////////////////////////////////////////////////////////////////////////////////////////////
// LiteNN: small fully connected networks evaluated per thread inside kernels.
//
// On GPU (Slang back end of kernel_slicer) 'LinearForward' becomes 'coopVecMatMulAdd' (cooperative vectors).
// The CPU implementation reproduces the GPU precision: weights, biases, the input and the result of every layer
// are fp16, accumulation is fp32. Intermediate arrays in user code are float.
//
// Weights of all layers (and of all networks) are stored in one 'std::vector<half>' member of the main class.
// One layer with 'inDim' inputs and 'outDim' outputs is a row-major matrix 'outDim x RowStride(inDim)'
// at 'matOffs' followed (not necessarily immediately) by 'outDim' biases at 'biasOffs'. Offsets are in elements
// of 'half'. Use 'AppendLayer' on the host to pack layers with the alignment the GPU needs:
//   matOffs  is a multiple of MATRIX_ALIGN (64 bytes),
//   biasOffs is a multiple of BIAS_ALIGN   (16 bytes),
//   rows are padded to ROW_ALIGN elements  (16 bytes).
//
// Rules for kernels:
//   1. 'a_inDim' and 'a_outDim' must be compile time constants (literal, constexpr, #define).
//   2. 'a_in' and 'a_out' are different arrays; their sizes are at least 'a_inDim' and 'a_outDim'.
////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <vector>
#include <cstdint>

#ifndef HALFFLOAT
#define HALFFLOAT
#endif
#include "LiteMath.h"

namespace LiteNN
{
  using LiteMath::half;

  static constexpr uint32_t MATRIX_ALIGN = 32; // in elements of half
  static constexpr uint32_t BIAS_ALIGN   = 8;  // in elements of half
  static constexpr uint32_t ROW_ALIGN    = 8;  // in elements of half

  static inline uint32_t RowStride(uint32_t a_inDim) { return (a_inDim + ROW_ALIGN - 1) / ROW_ALIGN * ROW_ALIGN; }

  static inline float RoundToHalf(float x) { return float(half(x)); }

  /**
  \brief a_out = W*a_in + b, W and b are taken from a_weights at a_matOffs and a_biasOffs
  */
  static inline void LinearForward(const half* a_weights, uint32_t a_matOffs, uint32_t a_biasOffs,
                                   const float* a_in, uint32_t a_inDim, float* a_out, uint32_t a_outDim)
  {
    float in[1024];
    for(uint32_t p = 0; p < a_inDim; p++)
      in[p] = RoundToHalf(a_in[p]);
    const uint32_t stride = RowStride(a_inDim);
    for(uint32_t j = 0; j < a_outDim; j++)
    {
      const half* row = a_weights + a_matOffs + j*stride;
      float acc = 0.0f;
      for(uint32_t p = 0; p < a_inDim; p++)
        acc += float(row[p])*in[p];
      a_out[j] = RoundToHalf(acc + float(a_weights[a_biasOffs + j]));
    }
  }

  static inline void ReLU(float* a_data, uint32_t a_size)
  {
    for(uint32_t i = 0; i < a_size; i++)
      a_data[i] = (a_data[i] > 0.0f) ? a_data[i] : 0.0f;
  }

  /**
  \brief a_out = max(W*a_in + b, 0)
  */
  static inline void LinearForward_ReLU(const half* a_weights, uint32_t a_matOffs, uint32_t a_biasOffs,
                                        const float* a_in, uint32_t a_inDim, float* a_out, uint32_t a_outDim)
  {
    LinearForward(a_weights, a_matOffs, a_biasOffs, a_in, a_inDim, a_out, a_outDim);
    ReLU(a_out, a_outDim);
  }

  ////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////// host side helpers /////////////////////////////////////////////
  ////////////////////////////////////////////////////////////////////////////////////////////////////////////

  struct LayerOffsets
  {
    uint32_t matOffs;
    uint32_t biasOffs;
  };

  /**
  \brief append one layer to a_weights with GPU alignment
  \param a_matrix -- row-major float matrix a_outDim x a_inDim
  \param a_bias   -- a_outDim floats
  */
  static inline LayerOffsets AppendLayer(std::vector<half>& a_weights, const float* a_matrix, const float* a_bias,
                                         uint32_t a_inDim, uint32_t a_outDim)
  {
    const uint32_t stride = RowStride(a_inDim);
    LayerOffsets res;
    res.matOffs  = uint32_t(a_weights.size() + MATRIX_ALIGN - 1) / MATRIX_ALIGN * MATRIX_ALIGN;
    res.biasOffs = (res.matOffs + a_outDim*stride + BIAS_ALIGN - 1) / BIAS_ALIGN * BIAS_ALIGN;
    a_weights.resize(res.biasOffs + a_outDim, half(0.0f));
    for(uint32_t j = 0; j < a_outDim; j++)
    {
      for(uint32_t p = 0; p < a_inDim; p++)
        a_weights[res.matOffs + j*stride + p] = half(a_matrix[j*a_inDim + p]);
      a_weights[res.biasOffs + j] = half(a_bias[j]);
    }
    return res;
  }
};
