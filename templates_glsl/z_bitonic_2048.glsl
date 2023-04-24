#version 460
#extension GL_GOOGLE_include_directive : require
#define SKIP_UBO_INCLUDE 1
#include "common{{Suffix}}.h"

layout(local_size_x = 1024, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, set = 0) buffer data0 { {{Type}} theArray[]; }; //

bool compare{{Lambda}}

layout( push_constant ) uniform kernelIntArgs
{
  int iNumElementsX;  
  int stage;
  int passOfStageBegin; 
  int a_invertModeOn;
} kgenArgs;

shared {{Type}} s_array[2048];

void main()
{
  const uint tid     = gl_GlobalInvocationID[0]; 
  const uint lid     = gl_LocalInvocationID[0];
  const uint blockId = (tid / 1024);

  s_array[lid + 0   ] = theArray[blockId * 2048 + lid + 0];
  s_array[lid + 1024] = theArray[blockId * 2048 + lid + 1024];

  barrier();

  for (int passOfStage = kgenArgs.passOfStageBegin; passOfStage >= 0; passOfStage--)
  {
    const int j = int(lid);
    const int r = int(1 << (passOfStage));
    const int lmask = r - 1;

    const int left = ((j >> passOfStage) << (passOfStage + 1)) + (j & lmask);
    const int right = left + r;

    const {{Type}} a = s_array[left];
    const {{Type}} b = s_array[right];

    const bool cmpRes = compare(a, b);

    const {{Type}} minElem = cmpRes ? a : b;
    const {{Type}} maxElem = cmpRes ? b : a;

    const int oddEven = int(tid) >> kgenArgs.stage; // (j >> stage)
    const bool isSwap = ((oddEven & 1) & kgenArgs.a_invertModeOn) != 0;

    const int minId = isSwap ? right : left;
    const int maxId = isSwap ? left : right;

    s_array[minId] = minElem;
    s_array[maxId] = maxElem;

    barrier();
  }

  theArray[blockId * 2048 + lid + 0]    = s_array[lid + 0];
  theArray[blockId * 2048 + lid + 1024] = s_array[lid + 1024];
}
