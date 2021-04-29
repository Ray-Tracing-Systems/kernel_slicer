#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include <vector>
#include <memory>

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

protected:
  unsigned int m_width;
  unsigned int m_height;
  std::vector<DataType> m_data;
};


#endif