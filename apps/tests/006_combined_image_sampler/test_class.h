#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>
#include "texture2d.h"



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
  

  std::shared_ptr<ITexture2DCombined> m_pCombinedImage = nullptr;

  int                 m_blurRadius;                  
  int                 m_width;
  int                 m_height;                
  int                 m_widthSmall;
  int                 m_heightSmall;
  float               m_gamma = 2.2F;
};

#endif