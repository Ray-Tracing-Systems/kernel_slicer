#ifndef BASIC_PROJ_LOGIC_H
#define BASIC_PROJ_LOGIC_H

#ifndef ISPC
#include "LiteMath.h"
#ifndef __OPENCL_VERSION__
using namespace LiteMath;
#endif
#endif

static inline uint RealColorToUint32_f3(float3 real_color)
{
  const float  r = real_color.x*255.0f;
  const float  g = real_color.y*255.0f;
  const float  b = real_color.z*255.0f;
  const uint8_t red   = (uint8_t)r;
  const uint8_t green = (uint8_t)g;
  const uint8_t blue  = (uint8_t)b;
  return red | (green << 8) | (blue << 16) | 0xFF000000;
}

static inline uint RealColorToUint32(float4 real_color)
{
  float  r = real_color.x*255.0f;
  float  g = real_color.y*255.0f;
  float  b = real_color.z*255.0f;
  float  a = real_color.w*255.0f;

  const uint8_t red   = (uint8_t)r;
  const uint8_t green = (uint8_t)g;
  const uint8_t blue  = (uint8_t)b;
  const uint8_t alpha = (uint8_t)a;

  return red | (green << 8) | (blue << 16) | (alpha << 24);
}

#define WIN_WIDTH  512
#define WIN_HEIGHT 512

static uint pitchOffset(uint x, uint y) { return y*WIN_WIDTH + x; } 

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif