#ifndef Denoise_UBO_H
#define Denoise_UBO_H

#include "OpenCLMath.h"

struct Denoise_UBO_Data
{
  float m_gamma;
  int m_size;
  unsigned int dummy_last;
};

#endif

