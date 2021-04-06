#ifndef ToneMapping_UBO_H
#define ToneMapping_UBO_H

#include "OpenCLMath.h"

struct ToneMapping_UBO_Data
{
  float m_gammaInv;
  unsigned int dummy_last;
};

#endif

