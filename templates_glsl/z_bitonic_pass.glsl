#version 460
#extension GL_GOOGLE_include_directive : require
#define SKIP_UBO_INCLUDE 1
#include "common{{Suffix}}.h"

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;
layout(binding = 0, set = 0) buffer data0 { {{Type}} theArray[]; }; //

bool compare{{Lambda}}

layout( push_constant ) uniform kernelIntArgs
{
  int iNumElementsX;  
  int stage;
  int passOfStage; 
  int a_invertModeOn;
} kgenArgs;

void main()
{
  const int j = int(gl_GlobalInvocationID[0]); 

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
