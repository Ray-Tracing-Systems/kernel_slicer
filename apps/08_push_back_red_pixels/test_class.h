#ifndef TEST_CLASS_H
#define TEST_CLASS_H

#include <vector>
#include <iostream>
#include <fstream>

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class RedPixels
{
public:

  typedef struct PixelInfoT
  {
    uint32_t value;
    uint32_t index;
  }PixelInfo;

  typedef uint32_t MyHandle;
  
  void SetMaxDataSize(size_t a_size); 
  void ProcessPixels(uint32_t* a_data, size_t a_dataSize);

  const std::vector<PixelInfo>& GetFoundPixels() const { return m_foundPixels; }
  const uint32_t                GetRedPixelsAmount() const { return m_redPixelsNum; }

//protected:
  
  void kernel1D_CountRedPixels(const uint32_t* a_data, size_t a_dataSize);
  void kernel1D_FindRedPixels(const uint32_t* a_data, size_t a_dataSize);
  void kernel1D_CopyPixels(const uint32_t* a_data, size_t a_dataSize, PixelInfo* a_outPixels);

  void kernel1D_PaintRedPixelsToYellow(uint32_t* a_data);

  uint32_t               m_size;
  uint32_t               m_redPixelsNum;
  uint32_t               m_otherPixelsNum;
  float                  m_testPixelsAmount = 0.0f;
  float                  m_testMin, m_testMax;
  std::vector<PixelInfo> m_foundPixels;
  //std::vector<PixelInfo> m_pixelsCopy;
};

#endif