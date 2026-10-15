#pragma once 
#include <cstdint>

static inline uint32_t part1by1_16(uint32_t n) {
    n &= 0x0000FFFFu;
    n = (n | (n << 8)) & 0x00FF00FFu;
    n = (n | (n << 4)) & 0x0F0F0F0Fu;
    n = (n | (n << 2)) & 0x33333333u;
    n = (n | (n << 1)) & 0x55555555u;
    return n;
}

static inline uint32_t morton2D_16(uint32_t x, uint32_t y) {
    return part1by1_16(x) | (part1by1_16(y) << 1);
}

static inline uint32_t compact1by1_16(uint32_t n) {
    n &= 0x55555555u;
    n = (n | (n >> 1)) & 0x33333333u;
    n = (n | (n >> 2)) & 0x0F0F0F0Fu;
    n = (n | (n >> 4)) & 0x00FF00FFu;
    n = (n | (n >> 8)) & 0x0000FFFFu;
    return n;
}

static inline void demorton2D_16(uint32_t m, uint32_t& x, uint32_t& y) {
    x = compact1by1_16(m);
    y = compact1by1_16(m >> 1);
}

#define OPTIMAL_LAYOUT_BLOCK_SIZE 4
static inline uint32_t optimal_layout_flatten(uint32_t x, uint32_t y) {
  const uint32_t blockSize = OPTIMAL_LAYOUT_BLOCK_SIZE;
  uint32_t blockX = x / blockSize;
  uint32_t blockY = y / blockSize;
  uint32_t blockIndex = morton2D_16(blockX, blockY);
  return blockIndex * blockSize * blockSize + (y % blockSize) * blockSize +
         x % blockSize;
}

static inline void optimal_layout_unflatten(uint32_t index,
                                            uint32_t &x, uint32_t &y) {
  uint32_t blockSize = OPTIMAL_LAYOUT_BLOCK_SIZE;
  const uint32_t blockArea = blockSize * blockSize;
  uint32_t blockIndex = index / blockArea;
  uint32_t inBlock = index % blockArea;

  uint32_t blockX, blockY;
  demorton2D_16(blockIndex, blockX, blockY);

  uint32_t px = inBlock % blockSize;
  uint32_t py = inBlock / blockSize;

  x = blockX * blockSize + px;
  y = blockY * blockSize + py;
}

static inline uint32_t optimal_layout_blocks_per_axis(uint32_t size) {
  const uint32_t blockSize = OPTIMAL_LAYOUT_BLOCK_SIZE;
  uint32_t blocks = (size + blockSize - 1) / blockSize;
  uint32_t pow2 = 1;
  while (pow2 < blocks)
    pow2 <<= 1;
  return pow2;
}

static inline uint32_t optimal_layout_layer_size(uint32_t width, uint32_t height) {
  const uint32_t bx = optimal_layout_blocks_per_axis(width);
  const uint32_t by = optimal_layout_blocks_per_axis(height);
  const uint32_t grid = bx > by ? bx : by;
  return grid * grid * OPTIMAL_LAYOUT_BLOCK_SIZE * OPTIMAL_LAYOUT_BLOCK_SIZE;
}
