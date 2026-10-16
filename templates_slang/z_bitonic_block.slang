#include "common{{Suffix}}_slang.h"

// sorts blocks of {{BlockSize}} elements in shared memory, {{BlockSize}}/2 threads per block

[[vk::binding(0, 0)]] RWStructuredBuffer<{{Type}}> theArray;

bool compare{{Lambda}}

struct KernelArgs
{
  int iNumElementsX;  
  int stage;
  int passOfStageBegin; 
  int a_invertModeOn;
};

static const uint HALF_BLOCK = {{BlockSize}}/2;

groupshared {{Type}} s_array[{{BlockSize}}];

[shader("compute")]
[numthreads({{BlockSize}}/2, 1, 1)]
void main(uint3 a_globalTID: SV_DispatchThreadID, uint3 a_localTID: SV_GroupThreadID, uniform KernelArgs kgenArgs)
{
  const uint tid     = a_globalTID[0]; 
  const uint lid     = a_localTID[0];
  const uint blockId = (tid / HALF_BLOCK);

  s_array[lid + 0]          = theArray[blockId*{{BlockSize}} + lid + 0];
  s_array[lid + HALF_BLOCK] = theArray[blockId*{{BlockSize}} + lid + HALF_BLOCK];
  GroupMemoryBarrierWithGroupSync();

  for (int passOfStage = kgenArgs.passOfStageBegin; passOfStage >= 0; passOfStage--)
  {
    const int j     = int(lid);
    const int r     = int(1 << (passOfStage));
    const int lmask = r - 1;

    const int left  = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const {{Type}} a = s_array[left];
    const {{Type}} b = s_array[right];

    const bool cmpRes = compare(a, b);

    const {{Type}} minElem = cmpRes ? a : b;
    const {{Type}} maxElem = cmpRes ? b : a;

    const int oddEven = int(tid) >> kgenArgs.stage; // (j >> stage)
    const bool isSwap = ((oddEven & 1) & kgenArgs.a_invertModeOn) != 0;

    const int minId = isSwap ? right : left;
    const int maxId = isSwap ? left  : right;

    s_array[minId] = minElem;
    s_array[maxId] = maxElem;
    GroupMemoryBarrierWithGroupSync();
  }

  theArray[blockId*{{BlockSize}} + lid + 0]          = s_array[lid + 0];
  theArray[blockId*{{BlockSize}} + lid + HALF_BLOCK] = s_array[lid + HALF_BLOCK];
}
