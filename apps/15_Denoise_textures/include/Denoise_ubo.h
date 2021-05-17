#ifndef Denoise_UBO_H
#define Denoise_UBO_H

#ifndef GLSL
#include "OpenCLMath.h"
#else
#define float4x4 mat4
#define float3   vec3
#define float4   vec4
#define uint32_t uint
#endif

struct Denoise_UBO_Data
{
  int m_linesDone;
  float m_noiseLevel;
  int m_sizeImg;
  float m_windowArea;
  uint m_normDepth_capacity;
  uint m_normDepth_size;
  uint m_texColor_capacity;
  uint m_texColor_size;
  uint dummy_last;
};

#endif

