#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#ifndef ISPC
#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif
#endif

static inline uint RealColorToUint32(float4 a_realColor, const float a_gamma)
{
  float  r = pow(clamp(a_realColor.x, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  g = pow(clamp(a_realColor.y, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  b = pow(clamp(a_realColor.z, 0.0F, 1.0F), a_gamma) * 255.0f;
  float  a = clamp(a_realColor.w, 0.0F, 1.0F) * 255.0f;

  const int red   = (int)r;
  const int green = (int)g;
  const int blue  = (int)b;
  const int alpha = (int)a;

  return (uint)(red | (green << 8) | (blue << 16) | (alpha << 24));
}

#define WIN_WIDTH  512
#define WIN_HEIGHT 512

static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

#endif

