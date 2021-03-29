#ifndef ToneMapping_UBO_H
#define ToneMapping_UBO_H

#include "OpenCLMath.h"

struct ToneMapping_UBO_Data
{
  int m_blurRadius;
  float m_gammaInv;
  int m_height;
  int m_heightSmall;
  int m_width;
  int m_widthSmall;
  unsigned int m_brightPixels_capacity;
  unsigned int m_brightPixels_size;
  unsigned int m_downsampledImage_capacity;
  unsigned int m_downsampledImage_size;
  unsigned int m_filterWeights_capacity;
  unsigned int m_filterWeights_size;
  unsigned int m_tempImage_capacity;
  unsigned int m_tempImage_size;
  unsigned int dummy_last;
};

#endif

