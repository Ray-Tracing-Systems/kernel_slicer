#include "common{{Suffix}}_slang.h"

[[vk::binding(0, 0)]] RWStructuredBuffer<{{Type}}> theArray;

bool compare{{Lambda}}

struct KernelArgs
{
  int iNumElementsX;  
  int stage;
  int passOfStage; 
  int a_invertModeOn;
};

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 a_globalTID: SV_DispatchThreadID, uniform KernelArgs kgenArgs)
{
  const int j = int(a_globalTID[0]); 

  const int r     = 1 << (kgenArgs.passOfStage);
  const int lmask = r - 1;

  const int left  = ((j >> kgenArgs.passOfStage) << (kgenArgs.passOfStage + 1)) + (j & lmask);
  const int right = left + r;

  const {{Type}} a = theArray[left];
  const {{Type}} b = theArray[right];

  const bool cmpRes = compare(a, b);

  const {{Type}} minElem = cmpRes ? a : b;
  const {{Type}} maxElem = cmpRes ? b : a;

  const int oddEven = j >> kgenArgs.stage;

  const bool isSwap = ((oddEven & 1) & kgenArgs.a_invertModeOn) != 0;

  const int minId = isSwap ? right : left;
  const int maxId = isSwap ? left  : right;

  theArray[minId] = minElem;
  theArray[maxId] = maxElem;
}
