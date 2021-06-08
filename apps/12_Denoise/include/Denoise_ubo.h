#ifndef Denoise_UBO_H
#define Denoise_UBO_H

#include "OpenCLMath.h"

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

