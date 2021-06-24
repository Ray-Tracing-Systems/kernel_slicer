#ifndef TEXTURE2D_H
#define TEXTURE2D_H

#include <vector>
#include <memory>
#include "aligned_alloc.h"
#include "sampler.h"

template<typename Type>
struct Texture2D
{
  Texture2D() : m_width(0), m_height(0) {}
  Texture2D(unsigned int w, unsigned int h) : m_width(w), m_height(h) { m_data.resize(w*h); }
  Texture2D(unsigned int w, unsigned int h, const Type* a_data) : m_width(w), m_height(h) 
  {
    m_data.resize(w*h);
    memcpy(m_data.data(), a_data, w*h*sizeof(Type));
  }
  
  void   resize(unsigned int width, unsigned int height) { m_width = width; m_height = height; m_data.resize(width*height); }
  float4 sample(const Sampler& a_sampler, float2 a_uv) const;    
  
  Type&  operator[](const int2 coord)       { return m_data[coord.y * m_width + coord.x]; }
  Type   operator[](const int2 coord) const { return m_data[coord.y * m_width + coord.x]; }

  unsigned int width() const  { return m_width; }
  unsigned int height() const { return m_height; }  
  unsigned int bpp() const    { return sizeof(Type); }
 
  const Type* getRawData() const { return m_data.data(); }

protected:
  float2 process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const;   

  unsigned int m_width;
  unsigned int m_height;
  cvex::vector<Type> m_data;  
};

//static inline unsigned int encodeNormal(float3 n)
//{
//  const int x = (int)(n.x*32767.0f);
//  const int y = (int)(n.y*32767.0f);
//
//  const unsigned int sign = (n.z >= 0) ? 0 : 1;
//  const unsigned int sx   = ((unsigned int)(x & 0xfffe) | sign);
//  const unsigned int sy   = ((unsigned int)(y & 0xffff) << 16);
//
//  return (sx | sy);
//}
//
//static inline float3 decodeNormal(unsigned int a_data)
//{  
//  const unsigned int a_enc_x = (a_data  & 0x0000FFFF);
//  const unsigned int a_enc_y = ((a_data & 0xFFFF0000) >> 16);
//  const float sign           = (a_enc_x & 0x0001) ? -1.0f : 1.0f;
//
//  const float x = ((short)(a_enc_x & 0xfffe))*(1.0f / 32767.0f);
//  const float y = ((short)(a_enc_y & 0xffff))*(1.0f / 32767.0f);
//  const float z = sign*sqrt(fmax(1.0f - x*x - y*y, 0.0f));
//
//  return make_float3(x, y, z);
//}

#endif
