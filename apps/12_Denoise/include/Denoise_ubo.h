#ifndef Denoise_UBO_H
#define Denoise_UBO_H

#include "OpenCLMath.h"

struct Denoise_UBO_Data
{
  int m_linesDone;
  float m_noiseLevel;
  int m_sizeImg;
  float m_windowArea;
  unsigned int m_normDepth_capacity;
  unsigned int m_normDepth_size;
  unsigned int m_texColor_capacity;
  unsigned int m_texColor_size;
  unsigned int dummy_last;
};

#endif

