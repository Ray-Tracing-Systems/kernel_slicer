#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>

#include "../14_filter_bloom_textures/include/BasicLogic.h" // We assume that all code that should pe passed to kernels will be just included both for CPU and OpenCL
#include "../14_filter_bloom_textures/sampler.h"
#include "../14_filter_bloom_textures/texture2d.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


class Denoise 
{
public:
  
  Denoise(){}
  Denoise(const int w, const int h);  
  
  void Resize(int w, int h); 

  // Non local mean denoise.
  void NLM_denoise(const int a_width, const int a_height, const float4* a_inImage, const Sampler& a_sampler, 
                   Texture2D<float4>& a_texture, unsigned int* a_outData1ui, const int32_t* a_inTexColor, 
                   const int32_t* a_inNormal, const float4* a_inDepth, const int a_windowRadius, 
                   const int a_blockRadius, const float a_noiseLevel);

protected:

  void kernel1D_PrepareData(const int32_t* a_inTexColor, const int32_t* a_inNormal, const float4* a_inDepth, 
                            const float4* a_inImage, Texture2D<float4>& a_texture);

  void kernel2D_GuidedTexNormDepthDenoise(const int a_width, const int a_height, const Sampler& a_sampler, 
                                          const Texture2D<float4>& a_texture, unsigned int* a_outData1ui, 
                                          const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);

  int                    m_width            = 0;
  int                    m_height           = 0;
  int                    m_sizeImg          = 0;
  int                    m_linesDone        = 0;

  Texture2D<float4> m_texColor;
  Texture2D<float4> m_normDepth;

  static constexpr float m_gamma            = 2.2F;

  float                  m_noiseLevel       = 0.1F;
  float                  m_windowArea       = 0.0F;
  static constexpr float m_gaussianSigma    = 1.0F / 50.0F;
  static constexpr float m_weightThreshold  = 0.03F;
  static constexpr float m_lerpCoefficeint  = 0.80F;
  static constexpr float m_counterThreshold = 0.05F; 
  static constexpr float m_DEG_TO_RAD       = 0.017453292519943295769236907684886F;
  static constexpr float m_fov              = m_DEG_TO_RAD * 90.0F;
};

#endif