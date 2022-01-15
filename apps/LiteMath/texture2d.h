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
  Texture2D(unsigned int w, unsigned int h) : m_width(w), m_height(h) { m_data.resize(w*h); m_fw = float(w); m_fh = float(h); }
  Texture2D(unsigned int w, unsigned int h, const Type* a_data) : m_width(w), m_height(h) 
  {
    m_data.resize(w*h);
    memcpy(m_data.data(), a_data, w*h*sizeof(Type));
    m_fw = float(w); m_fh = float(h);
  }
  
  void   resize(unsigned int width, unsigned int height) { m_width = width; m_height = height; m_data.resize(width*height); m_fw = float(width); m_fh = float(height); }
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
  float m_fw;
  float m_fh;
  cvex::vector<Type> m_data;  
};

/**
  \brief combined image of unknown type (could be any) and sampler
*/
struct ITexture2DCombined 
{
  virtual float4       sample(float2 a_uv) const = 0;  
  
  virtual unsigned int width()             const = 0;
  virtual unsigned int height()            const = 0;
  virtual unsigned int bpp()               const = 0;
  virtual unsigned int format()            const = 0; ///<! return uint(VkFormat) value 
  
  virtual Sampler      getSampler()        const = 0;
  virtual const void*  getRawData()        const = 0;
};

template<typename T> uint32_t GetVulkanFormat();

/**
  \brief Can have specific effitient implementations for various textures and samplers
*/
template<typename Type>
std::shared_ptr<ITexture2DCombined> MakeCombinedTexture2D(std::shared_ptr<Texture2D<Type> > a_texture, Sampler a_sampler) 
{ 
  // general implementation
  //
  struct CobminedImageSampler : public ITexture2DCombined
  {
    float4 sample(float2 a_uv) const override { return m_pTexture->sample(m_sampler, a_uv); }  
  
    unsigned int width()             const override { return m_pTexture->width();  }
    unsigned int height()            const override { return m_pTexture->height(); }
    unsigned int bpp()               const override { return m_pTexture->bpp();    }
    unsigned int format()            const override { return GetVulkanFormat<Type>();  } 
    Sampler      getSampler()        const override { return m_sampler;                }
    const void*  getRawData()        const override { return m_pTexture->getRawData(); }

    std::shared_ptr<Texture2D<Type> > m_pTexture;
    Sampler                           m_sampler;
  };

  std::shared_ptr<CobminedImageSampler> pObj = std::make_shared<CobminedImageSampler>();
  pObj->m_pTexture = a_texture;
  pObj->m_sampler  = a_sampler;
  return pObj;
}


#endif
