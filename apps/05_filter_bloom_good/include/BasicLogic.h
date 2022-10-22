#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#ifndef ISPC
#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif
#endif

static inline uint RealColorToUint32(float4 real_color)
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

static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif