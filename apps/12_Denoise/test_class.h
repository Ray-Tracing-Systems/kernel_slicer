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
  Denoise(const int w, const int h);  
  
  void Resize(int w, int h); 

  // Non local mean denoise.
  virtual void NLM_denoise(const int a_width, const int a_height, 
                           const float4* a_inImage     __attribute__((size("a_width", "a_height"))), 
                           unsigned int* a_outData1ui  __attribute__((size("a_width", "a_height"))), 
                           const int32_t* a_inTexColor __attribute__((size("a_width", "a_height"))), 
                           const int32_t* a_inNormal   __attribute__((size("a_width", "a_height"))), 
                           const float4* a_inDepth     __attribute__((size("a_width", "a_height"))), 
                           const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel1D_int32toFloat4(const int32_t* a_inTexColor, const int32_t* a_inNormal, const float4* a_inDepth);
  void kernel2D_GuidedTexNormDepthDenoise(const int a_width, const int a_height, const float4* a_inImage, unsigned int* a_outData1ui,
                                          const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel);

  int                    m_width            = 0;
  int                    m_height           = 0;
  int                    m_sizeImg          = 0;
  int                    m_linesDone        = 0;

  std::vector<float4> m_texColor;
  std::vector<float4> m_normDepth;

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