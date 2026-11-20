#include "test_class.h"

void TiledGemm2D::MatMul(const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  kernelBE2D_MatMul<TILE, TILE>(DivUp(a_N, TILE), DivUp(a_M, TILE), a_A, a_B, a_C, a_M, a_N, a_K);
}

void TiledGemm2D::MatMulNaive(const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  kernel2D_MatMulNaive(a_M, a_N, a_A, a_B, a_C, a_K);
}

// A 2D block (blockX, blockY) computes a tile of C; the thread (x, y) of the block computes its element (row0 + y, col0 + x).
// Loop over K: tiles of A and B are loaded to shared memory, then every thread adds products of a row of the tile of A
// and a column of the tile of B. Tiles are square: bsizeX == bsizeY == TILE.
// Loops are written for CPU (y is the outer loop, x the inner one); '-reorderLoops YX' in kmake.json makes x the fast index on GPU.
//
template<uint bsizeX, uint bsizeY>
void TiledGemm2D::kernelBE2D_MatMul(uint a_blocksX, uint a_blocksY, const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  for(uint blockY = 0; blockY < a_blocksY; blockY++)
  {
    for(uint blockX = 0; blockX < a_blocksX; blockX++)
    {
      static_assert(bsizeX == bsizeY, "tiles of A and B are square");

      float tileA[bsizeY][bsizeX];                        // shared memory of the block
      float tileB[bsizeY][bsizeX];
      float acc[bsizeY][bsizeX];                          // accumulators, one per thread

      const uint row0 = blockY*bsizeY;
      const uint col0 = blockX*bsizeX;

      for(uint y = 0; y < bsizeY; y++) [[parallel]]       // 2D parallel loop: thread (x, y) of the block
        for(uint x = 0; x < bsizeX; x++) [[parallel]]
          acc[y][x] = 0.0f;

      for(uint k0 = 0; k0 < a_K; k0 += bsizeX)
      {
        for(uint y = 0; y < bsizeY; y++) [[parallel]]     // load tiles, zero outside of the matrices
        {
          for(uint x = 0; x < bsizeX; x++) [[parallel]]
          {
            tileA[y][x] = (row0 + y < a_M && k0 + x < a_K) ? a_A[(row0 + y)*a_K + k0 + x] : 0.0f;
            tileB[y][x] = (k0 + y < a_K && col0 + x < a_N) ? a_B[(k0 + y)*a_N + col0 + x] : 0.0f;
          }
        }

        for(uint y = 0; y < bsizeY; y++) [[parallel]]
        {
          for(uint x = 0; x < bsizeX; x++) [[parallel]]
          {
            float sum = acc[y][x];
            for(uint k = 0; k < bsizeX; k++)
              sum += tileA[y][k]*tileB[k][x];
            acc[y][x] = sum;
          }
        }
      }

      for(uint y = 0; y < bsizeY; y++) [[parallel]]
      {
        for(uint x = 0; x < bsizeX; x++) [[parallel]]
        {
          if(row0 + y < a_M && col0 + x < a_N)
            a_C[(row0 + y)*a_N + col0 + x] = acc[y][x];
        }
      }
    }
  }
}

void TiledGemm2D::kernel2D_MatMulNaive(uint a_M, uint a_N, const float* a_A, const float* a_B, float* a_C, uint a_K)
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
