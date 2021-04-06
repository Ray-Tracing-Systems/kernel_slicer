#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include "include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL

#include <vector>
#include <iostream>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Denoise 
{
public:

  Denoise(){}
  
  void SetMaxImageSize(int w, int h);

  // Non local mean denoise.
  void NLM_denoise(int a_width, const int a_height, const float4* a_inImage, unsigned int* a_outData1ui, const int32_t* a_inTexColor, 
const int32_t* a_inNormal, const float* a_inDepth, const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);

protected:

  void kernel1D_int32toFloat4(const int32_t* a_inTexColor, const int32_t* a_inNormal, const float* a_inDepth, float4* a_texColor, float4* a_normDepth);

  void kernel2D_GuidedTexNormDepthDenoise(int a_width, const int a_height, const float4* a_inImage, const float4* a_inTexColor, 
  const float4* a_inNormDepth, unsigned int* a_outData1ui, const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);

  int   m_width;
  int   m_height;
  int   m_size;
  float m_gammaInv = 1.0f / 2.2f;
};

#endif