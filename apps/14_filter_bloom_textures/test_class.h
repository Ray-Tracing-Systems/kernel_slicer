#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>
#include "texture2d.h"

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static std::vector<float> createGaussKernelWeights1D_HDRImage(int size, float a_sigma)
{
  std::vector<float> gKernel;
  gKernel.resize(size);
  // set standard deviation to 1.0
  //
  float sigma = a_sigma;
  float s = 2.0f * sigma * sigma;
  // sum is for normalization
  float sum = 0.0;
  int halfSize = size / 2;
  for (int x = -halfSize; x <= halfSize; x++)
  {
    float r = sqrtf((float)(x*x));
    int index = x + halfSize;
    gKernel[index] = (exp(-(r) / s)) / (3.141592654f * s);
    sum += gKernel[index];
  }
  // normalize the Kernel
  for (int i = 0; i < size; ++i)
    gKernel[i] /= sum;
  return gKernel;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class ToneMapping 
{
public:
  
  ToneMapping(){}
  ToneMapping(const int w, const int h);  
  void SetSize(const int w, const int h);

  virtual void Bloom (const int a_width, const int a_height, const Texture2D<float4>& a_texture2d, 
                      unsigned int* outData1ui __attribute__((size("a_width", "a_height"))) );

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel2D_ExtractBrightPixels(const int a_width, const int a_height, const Texture2D<float4>& a_texture2d,        Texture2D<float4>& a_brightPixels);
  void kernel2D_DownSample4x       (const int a_width, const int a_height, const Texture2D<float4>& a_texture2dFullRes, Texture2D<float4>& a_dataSmallRes);
  void kernel2D_BlurX              (const int a_width, const int a_height, const Texture2D<float4>& a_texture2d,        Texture2D<float4>& a_dataOut);
  void kernel2D_BlurY              (const int a_width, const int a_height, const Texture2D<float4>& a_texture2d,        Texture2D<float4>& a_dataOut);
  void kernel2D_MixAndToneMap      (const int a_width, const int a_height, const Texture2D<float4>& a_texture2d, unsigned int* outData1ui);

  std::vector<float>  m_filterWeights;
  Texture2D<float4>   m_brightPixels;
  Texture2D<float4>   m_downsampledImage;
  Texture2D<float4>   m_tempImage;
  Sampler             m_sampler;  

  int                 m_blurRadius;                  
  int                 m_width;
  int                 m_height;                
  int                 m_widthSmall;
  int                 m_heightSmall;
  float               m_gamma = 2.2F;
};

#endif