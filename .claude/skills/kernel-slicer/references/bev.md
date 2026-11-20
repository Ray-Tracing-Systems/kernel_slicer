# BEV pattern (Block Expansion Vectorization)

Experimental. Back ends: Slang, GLSL, ISPC (`LiteNN` block functions: Slang and ISPC only).
Use BEV when the threads of a work group must cooperate through shared memory. For ordinary data-parallel work
use IPV: it is simpler and as fast.

## Anatomy of a BEV kernel

- Name prefix `kernelBE1D_` or `kernelBE2D_`; the kernel is a member function template on the block size:
  `template<uint bsize>` (1D) or `template<uint bsizeX, uint bsizeY>` (2D). It is not virtual (it can't be).
- The body is one loop over blocks (two nested loops for 2D), like the loops of an IPV kernel. One iteration is a block,
  one block is one work group of `bsize` (`bsizeX*bsizeY`) threads.
- The control function calls it with the block size as template argument; a named constant may be used:
  `kernelBE1D_Reduce<BLOCK_SIZE>(numBlocks, ...)`. Loops over blocks are bounded by arguments: the numbers of blocks.

Who executes the code of a block:

| Statement in the block | Executed by | Notes |
|------------------------|-------------|-------|
| array declared at the top level of the block | — | shared memory of the group; no initializer |
| `for(T i = start; cond; i++) [[parallel]]` | all threads, thread `t` takes iterations `t`, `t + bsize`, ... | barriers before and after |
| two nested `[[parallel]]` loops (2D kernels only) | thread `(x, y)` of the group | dimensions as for `kernel2D_` and `-reorderLoops` |
| ordinary `for`/`while`/`if` with `[[parallel]]` inside | all threads | condition must be the same for all threads |
| other statement that writes memory | one thread | barrier after it if it writes shared memory |
| other statement (local variables, reading memory) | every thread, own copy of variables | |
| `LiteNN::LinearForwardBlock(...)` | all threads together | separate statement, not in a parallel loop |
| `static_assert(...)` | — | checked by the C++ compiler only |

"Writes memory": assignment to an array of the block, a buffer (`a_out[i] = ...`), a class member, a call of a
non-const method or of a function taking a non-const pointer or reference.

## Example: block reduction with three levels (`apps/tests/092_bev_block_reduce`)

```cpp
static constexpr uint BLOCK_SIZE = 256;

void BlockReduce::Reduce(const int* a_in, int* a_out, uint a_size)   // a_in [[size("a_size")]], a_out [[size("1")]]
{
  const uint blocks1 = (a_size  + BLOCK_SIZE - 1)/BLOCK_SIZE;
  const uint blocks2 = (blocks1 + BLOCK_SIZE - 1)/BLOCK_SIZE;
  kernelBE1D_ReduceBlocks<BLOCK_SIZE>(blocks1, a_in,           m_sums1.data(), a_size);
  kernelBE1D_ReduceBlocks<BLOCK_SIZE>(blocks2, m_sums1.data(), m_sums2.data(), blocks1);
  kernelBE1D_ReduceBlocks<BLOCK_SIZE>(1,       m_sums2.data(), a_out,          blocks2);
}

template<uint bsize>
void BlockReduce::kernelBE1D_ReduceBlocks(uint a_blockNum, const int* a_in, int* a_out, uint a_size)
{
  for(uint blockId = 0; blockId < a_blockNum; blockId++)
  {
    int sdata[bsize];                                     // shared memory

    for(uint i = 0; i < bsize; i++) [[parallel]]
    {
      const uint globalId = blockId*bsize + i;
      sdata[i] = (globalId < a_size) ? a_in[globalId] : 0;
    }

    for(uint stride = bsize/2; stride > 0; stride /= 2)   // ordinary loop: all threads, barriers inside
    {
      for(uint i = 0; i < stride; i++) [[parallel]]
        sdata[i] += sdata[i + stride];
    }

    a_out[blockId] = sdata[0];                            // one thread
  }
}
```

Member vectors used as intermediate buffers (`m_sums1`, `m_sums2`) are resized in the constructor, before
`CommitDeviceData()`. A variable of the block may change in an ordinary loop and be read in a parallel loop
(`uint src = 0; ... src = bsize - src;` in `apps/tests/093_bev_block_scan`): every thread has its own copy.

## 2D blocks (`apps/tests/096_bev_tiled_gemm_2d`)

```cpp
template<uint bsizeX, uint bsizeY>
void TiledGemm2D::kernelBE2D_MatMul(uint a_blocksX, uint a_blocksY, const float* a_A, const float* a_B, float* a_C, uint a_M, uint a_N, uint a_K)
{
  for(uint blockY = 0; blockY < a_blocksY; blockY++)
  {
    for(uint blockX = 0; blockX < a_blocksX; blockX++)
    {
      static_assert(bsizeX == bsizeY, "tiles of A and B are square");
      float tileA[bsizeY][bsizeX];                        // 2D arrays of the block
      float tileB[bsizeY][bsizeX];
      float acc[bsizeY][bsizeX];                          // one accumulator per thread
      ...
      for(uint k0 = 0; k0 < a_K; k0 += bsizeX)
      {
        for(uint y = 0; y < bsizeY; y++) [[parallel]]     // 2D parallel loop
          for(uint x = 0; x < bsizeX; x++) [[parallel]]
          {
            tileA[y][x] = ...;
            tileB[y][x] = ...;
          }
        ...
      }
    }
  }
}
// control function: kernelBE2D_MatMul<TILE, TILE>(DivUp(a_N, TILE), DivUp(a_M, TILE), ...);
```

Write `y` loops outside and `x` loops inside, as for the CPU, and put `"-reorderLoops": "YX"` into `kmake.json`
(the same rule as for `kernel2D_`). A single `[[parallel]]` loop in a 2D block is distributed over all threads.
On ISPC 2D blocks are much faster than 1D blocks with `i / TILE`, `i % TILE` indexing (`y` is uniform, `x` is
contiguous over the gang).

## Block level functions: `LiteNN::LinearForwardBlock` (`apps/tests/095_bev_mlp_batch`)

```cpp
half  act[BATCH*NN_HIDDEN_DIM];                           // inputs of a layer, fp16, row b is vector b of the batch
float res[BATCH*NN_HIDDEN_DIM];                           // outputs of a layer, fp32
...
LiteNN::LinearForwardBlock(m_weights.data(), m_matOffs[0], m_biasOffs[0], act, NN_INPUT_DIM, res, NN_HIDDEN_DIM, BATCH);
for(uint i = 0; i < BATCH*NN_HIDDEN_DIM; i++) [[parallel]]
  act[i] = half(std::max(res[i], 0.0f));                  // ReLU and conversion: a parallel loop
```

Rules: weights are `m_weights.data()` of a `std::vector<half>` member packed by `LiteNN::AppendLayer`, with offsets
(the table of layers is not supported yet); `in` and `out` are different arrays of the block; sizes are compile time
constants, multiples of 16. Add `"-coop_matrix": 1` to `options` of `kmake.json` to use cooperative matrices
(`VK_KHR_cooperative_matrix`, Slang); then create the device with `<MainClass>_Generated_ListRequiredDeviceFeatures()`
and `vk_utils::globalContextInit(features, ...)`. Without the option it is an ordinary loop that works on any GPU.

## Atomics in shared memory (`apps/tests/097_bev_histogram`)

```cpp
uint hist[NUM_BINS];                                      // histogram of the block, shared memory
for(uint i = 0; i < NUM_BINS; i++) [[parallel]]
  hist[i] = 0;
for(uint i = 0; i < ELEMS_PER_BLOCK; i++) [[parallel]]
{
  const uint globalId = blockId*ELEMS_PER_BLOCK + i;
  if(globalId < a_size)
    InterlockedAdd(hist[a_in[globalId] % NUM_BINS], 1u); // atomic in shared memory
}
for(uint i = 0; i < NUM_BINS; i++) [[parallel]]
  if(hist[i] != 0)
    InterlockedAdd(a_hist[i], hist[i]);                  // atomic in global memory: histograms of blocks are added
```

`LiteMath::InterlockedAdd(mem, value)` and `InterlockedAdd(mem, value, oldValue)` work on arrays of the block and on
buffers. The buffer is cleared by an IPV kernel called before the BEV kernel in the same control function.

## Pitfalls

1. **A statement that writes memory and changes a variable of the block** is an error (it runs on one thread, the
   copies of the variable in other threads would differ):
   ```cpp
   for(uint i = 0; i < 4; i++) { a_out[i] = sdata[i]; count++; }   // error
   ```
   Split it: compute `count` in its own statement, write memory in another.
2. **The condition of an ordinary loop or `if` that contains `[[parallel]]`** must not depend on variables of a
   parallel loop: all threads must reach the same barriers.
3. **`[[parallel]]` inside `[[parallel]]`** is allowed only as a 2D parallel loop of `kernelBE2D_`, perfectly nested.
4. **Barriers synchronize shared memory only.** Do not write a buffer in one thread of a group and read it from other
   threads of the same group in the same kernel; use another kernel call.
5. **Arrays of the block can't be initialized** (`int s[bsize] = {};` is an error): fill them in a parallel loop.
6. **Other kernels of the class stay `virtual`** (ISPC back end overrides them); only the BEV template kernel is not.
7. **ISPC evaluates both sides of `?:` with a varying condition** under a mask, and the address of a masked vector
   load must still be valid. So the translator makes unsigned indexes with subtraction in arrays of the block signed
   (`c ? a[i - 1] : 0` with `i == 0` would otherwise address far outside of the array).

## Verification

Same as for any parallel primitive: compare with the CPU result on many sizes, including block borders
(`1, bsize - 1, bsize, bsize + 1, ...`), the maximum size and random sizes, and run many times (races between
threads of a group or between groups show up only there). Test the Slang, GLSL and ISPC variants
(`-ispc_threads 0/1/2`) listed in `ktest.json`.
