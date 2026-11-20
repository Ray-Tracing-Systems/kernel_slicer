#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
using LiteMath::uint;

static constexpr uint TILE = 16;                               // tile of C is computed by a 2D block of TILE x TILE threads

static inline uint DivUp(uint a, uint b) { return (a + b - 1)/b; }

// C[M x N] = A[M x K] * B[K x N], all matrices are row-major; the same algorithm as in test 094, but with a 2D block:
// 'MatMul' -- tiles of A and B in shared memory (block expansion 2D), 'MatMulNaive' -- one thread per element of C (IPV).
//
class TiledGemm2D
{
public:
  TiledGemm2D(){}

  virtual void MatMul     (const float* a_A [[size("a_M*a_K")]], const float* a_B [[size("a_K*a_N")]], float* a_C [[size("a_M*a_N")]], uint a_M, uint a_N, uint a_K);
  virtual void MatMulNaive(const float* a_A [[size("a_M*a_K")]], const float* a_B [[size("a_K*a_N")]], float* a_C [[size("a_M*a_N")]], uint a_M, uint a_N, uint a_K);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  template<uint bsizeX, uint bsizeY>
  void kernelBE2D_MatMul(uint a_blocksX, uint a_blocksY, const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K);
  virtual void kernel2D_MatMulNaive(uint a_M, uint a_N, const float* a_A, const float* a_B, float* a_C, uint a_K);
};
