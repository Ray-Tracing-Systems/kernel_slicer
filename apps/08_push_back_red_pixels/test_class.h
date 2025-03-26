#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <cstdint>
#include <vector>
#include <iostream>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RedPixels
{
public:

  RedPixels(){}

  struct PixelInfo
  {
    uint32_t value;
    uint32_t index;
  };

  typedef uint32_t MyHandle;
  
  void SetMaxDataSize(size_t a_size); 
  virtual void ProcessPixels(const uint32_t* a_inData  [[size("a_dataSize")]],
                                   uint32_t* a_outData [[size("a_dataSize")]], size_t a_dataSize);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

//protected:
  
  virtual void kernel1D_CountRedPixels(const uint32_t* a_data, size_t a_dataSize);
  virtual void kernel1D_FindRedPixels(const uint32_t* a_data, size_t a_dataSize);
  
  virtual void kernel1D_CopyPixels(const uint32_t* a_data, size_t a_dataSize, PixelInfo* a_outPixels);
  virtual void kernel1D_PaintRedPixelsToYellow(uint32_t* a_data);

  uint32_t               m_size;
  uint32_t               m_redPixelsNum;
  uint32_t               m_otherPixelsNum;
  float                  m_testPixelsAmount = 0.0f;
  float                  m_testMin, m_testMax;
  std::vector<PixelInfo> m_foundPixels;
  //std::vector<PixelInfo> m_pixelsCopy;
};

#endif