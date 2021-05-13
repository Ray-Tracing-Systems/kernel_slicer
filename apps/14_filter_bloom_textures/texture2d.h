#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include <vector>
#include <memory>
#include "sampler.h"

///////////////////////////////////////////////////////////////////////


template<typename DataType>
struct Texture2D
{
  Texture2D() : m_width(0), m_height(0) {}
  Texture2D(unsigned int w, unsigned int h) : m_width(w), m_height(h) { m_data.resize(w*h); }
  Texture2D(unsigned int w, unsigned int h, const DataType* a_data) : m_width(w), m_height(h) 
  {
    m_data.resize(w*h);
    memcpy(m_data.data(), a_data, w*h*sizeof(DataType));
  }
  
  float2   process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const; 
  DataType sample(const Sampler* a_sampler, float2 a_uv) const;    
  void     write_pixel(const uint posPixel, const DataType color) { m_data[posPixel] = color; }


protected:
  unsigned int m_width;
  unsigned int m_height;
  std::vector<DataType> m_data;
};


float2 get_uv(const int x, const int y, const uint width, const uint height);


#endif
