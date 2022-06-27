#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>

#include "Image2d.h"
using LiteImage::ICombinedImageSampler;
using LiteMath::uint;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class TestCombinedImage 
{
public:

  TestCombinedImage();  

  virtual void Run(const int a_width, const int a_height, unsigned int* outData1ui __attribute__((size("a_width", "a_height"))) );

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class    

protected:

  void kernel2D_Run(const int a_width, const int a_height, unsigned int* outData1ui);
  
  std::vector< std::shared_ptr<ICombinedImageSampler> > m_textures;
  std::vector< const ICombinedImageSampler* >           m_textures2;

  int                 m_blurRadius;                  
  int                 m_width;
  int                 m_height;                
  int                 m_widthSmall;
  int                 m_heightSmall;
  float               m_gamma = 2.2F;

  static constexpr uint FRACTAL_ITERATIONS = 256;
};

#endif