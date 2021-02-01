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

  struct PixelInfo
  {
    uint32_t value;
    uint32_t index;
  };
  
  void SetMaxDataSize(size_t a_size); 
  void ProcessPixels(const uint32_t* a_data, size_t a_dataSize);

  const std::vector<PixelInfo>& GetFoundPixels() const { return m_foundPixels; }
  const uint32_t                GetRedPixelsAmount() const { return m_redPixelsNum; }

protected:
  
  void kernel1D_CountRedPixels(const uint32_t* a_data, size_t a_dataSize);
  void kernel1D_FindRedPixels(const uint32_t* a_data, size_t a_dataSize);

  uint32_t               m_size;
  uint32_t               m_redPixelsNum;
  std::vector<PixelInfo> m_foundPixels;

};

#endif