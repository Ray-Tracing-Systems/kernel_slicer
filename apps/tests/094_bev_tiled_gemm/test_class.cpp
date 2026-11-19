#include "test_class.h"

void TiledGemm::MatMul(const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  const uint blocks = DivUp(a_M, TILE)*DivUp(a_N, TILE);
  kernelBE1D_MatMul<TILE*TILE>(blocks, a_A, a_B, a_C, a_M, a_N, a_K);
}

void TiledGemm::MatMulNaive(const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  kernel2D_MatMulNaive(a_M, a_N, a_A, a_B, a_C, a_K);
}

// One block computes a tile TILE x TILE of C. Loop over K: tiles of A and B are loaded to shared memory,
// then every thread adds products of a row of the tile of A and a column of the tile of B to its element of C.
//
template<uint bsize>
void TiledGemm::kernelBE1D_MatMul(uint a_blockNum, const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    float tileA[TILE*TILE];                               // shared memory of the block
    float tileB[TILE*TILE];
    float acc[bsize];                                     // accumulators, one per thread (bsize == TILE*TILE)

    const uint tilesX = DivUp(a_N, TILE);                 // helper function in the code of the block
    const uint row0   = (blockId / tilesX)*TILE;
    const uint col0   = (blockId % tilesX)*TILE;

    for(uint i = 0; i < bsize; i++) [[parallel]]
      acc[i] = 0.0f;

    for(uint k0 = 0; k0 < a_K; k0 += TILE)
    {
      for(uint i = 0; i < bsize; i++) [[parallel]]        // load tiles, zero outside of the matrices
      {
        const uint y = i / TILE;
        const uint x = i % TILE;
        tileA[i] = (row0 + y < a_M && k0 + x < a_K) ? a_A[(row0 + y)*a_K + k0 + x] : 0.0f;
        tileB[i] = (k0 + y < a_K && col0 + x < a_N) ? a_B[(k0 + y)*a_N + col0 + x] : 0.0f;
      }

      for(uint i = 0; i < bsize; i++) [[parallel]]
      {
        const uint y = i / TILE;
        const uint x = i % TILE;
        float sum = acc[i];
        for(uint k = 0; k < TILE; k++)
          sum += tileA[y*TILE + k]*tileB[k*TILE + x];
        acc[i] = sum;
      }
    }

    for(uint i = 0; i < bsize; i++) [[parallel]]
    {
      const uint y = i / TILE;
      const uint x = i % TILE;
      if(row0 + y < a_M && col0 + x < a_N)
        a_C[(row0 + y)*a_N + col0 + x] = acc[i];
    }
  }
}

void TiledGemm::kernel2D_MatMulNaive(uint a_M, uint a_N, const float* a_A, const float* a_B, float* a_C, uint a_K)
{
  for(uint y = 0; y < a_M; y++)
  {
    for(uint x = 0; x < a_N; x++)
    {
      float sum = 0.0f;
      for(uint k = 0; k < a_K; k++)
        sum += a_A[y*a_K + k]*a_B[k*a_N + x];
      a_C[y*a_N + x] = sum;
    }
  }
}
