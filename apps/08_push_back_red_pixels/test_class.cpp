#include "test_class.h"
#include "Bitmap.h"
#include <cassert>
#include <algorithm>
#include <memory>
#include <cstring>

inline bool PixelIsRed(uint32_t a_pixelValue)
{
  const uint32_t red   = (a_pixelValue & 0x000000FF);
  const uint32_t green = (a_pixelValue & 0x0000FF00) >> 8;
  const uint32_t blue  = (a_pixelValue & 0x00FF0000) >> 16;
  return (red >= 250) && (green < 5) && (blue < 5);
}

void RedPixels::SetMaxDataSize(size_t a_size)
{
  m_size         = uint32_t(a_size);
  m_redPixelsNum = 0;
  m_foundPixels.reserve(a_size);
  //m_pixelsCopy.resize(a_size);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void RedPixels::kernel1D_CountRedPixels(const uint32_t* a_data, size_t a_dataSize)
{
  m_redPixelsNum     = 0;
  m_otherPixelsNum   = 0;
  m_testPixelsAmount = 0.0f;
  m_testMin          = +100000000.0f;
  m_testMax          = -100000000.0f;

  for(uint32_t i = 0; i<a_dataSize; i++)
  {
    if(PixelIsRed(a_data[i]))
    {
      m_redPixelsNum++;
      m_testPixelsAmount -= 0.5f; // for example some function which define how pixel is red more precicely
    }
    else
    {
      float strangeVal = ((float)a_data[i])*((float)i);
      if(i == 1000)
        strangeVal = -10.0f;
      m_testMin = std::min(m_testMin, strangeVal);
      m_testMax = std::max(m_testMax, strangeVal);
      m_otherPixelsNum++;
    }
  }
}

void RedPixels::kernel1D_FindRedPixels(const uint32_t* a_data, size_t a_dataSize)
{
  m_foundPixels.resize(0);
  for(uint32_t i = 0; i<a_dataSize; i++)
  {
    const uint32_t pixValue = a_data[i];
    if(PixelIsRed(pixValue))
    {
      PixelInfo info;
      info.index = i;
      info.value = pixValue;
      m_foundPixels.push_back(info);
    }
  }
}

void RedPixels::kernel1D_PaintRedPixelsToYellow(uint32_t* a_data)
{
  for(uint32_t pixelId = 0; pixelId < m_foundPixels.size(); pixelId++)
    a_data[m_foundPixels[pixelId].index] = 0x0000FFFF;
}

void RedPixels::kernel1D_CopyPixels(const uint32_t* a_data, size_t a_dataSize, PixelInfo* a_outPixels)
{
  for(uint32_t i = 0; i<a_dataSize; i++)
  {
    PixelInfo info;
    info.index     = i;
    info.value     = a_data[i];
    a_outPixels[i] = info;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Cont, typename Pred>
Cont filter(const Cont &container, Pred predicate) 
{
  Cont result;
  std::copy_if(container.begin(), container.end(), std::back_inserter(result), predicate);
  return result;
}

void RedPixels::ProcessPixels(const uint32_t* a_inData, uint32_t* a_outData, size_t a_dataSize)
{
  kernel1D_CountRedPixels(a_inData, a_dataSize);
  kernel1D_FindRedPixels (a_inData, a_dataSize);
  memcpy(a_outData, a_inData, a_dataSize*sizeof(uint32_t));
  kernel1D_PaintRedPixelsToYellow(a_outData);
}