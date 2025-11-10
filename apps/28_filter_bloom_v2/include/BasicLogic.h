#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#ifndef ISPC
#include "LiteMath.h"
#ifndef CUDA_MATH
using namespace LiteMath;
#endif
#endif

#if defined(USE_CUDA)
#define _HostDevice_ __host__ __device__
#elif defined(__HIPCC__)
#define _HostDevice_ __host__ __device__
#else
#define _HostDevice_ 
#endif 

_HostDevice_ static inline uint RealColorToUint32(float4 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  float  a = real_color.w*255.0f;

  const int red   = (int)r;
  const int green = (int)g;
  const int blue  = (int)b;
  const int alpha = (int)a;

  return (uint)(red | (green << 8) | (blue << 16) | (alpha << 24));
}

#define WIN_WIDTH  512
#define WIN_HEIGHT 512

_HostDevice_ static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif