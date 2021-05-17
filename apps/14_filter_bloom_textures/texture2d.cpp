#include "texture2d.h"


template<> float4 Texture2D<float4>::sample(const Sampler* a_sampler, float2 a_uv) const
{
  bool useBorderColor = false;

  a_uv = process_coord(a_sampler->m_addressU, a_uv, &useBorderColor);
    
  if (useBorderColor) {
    return a_sampler->m_borderColor;
  }

  const float2 textureSize = make_float2(m_width, m_height);
  const float2 scaledUV    = textureSize * a_uv;
  const int2   baseTexel   = make_int2(scaledUV.x, scaledUV.y);
  const int    stride      = m_width;

  if (a_sampler->m_filter == Sampler::Filter::MIN_MAG_MIP_POINT) {
    return m_data[pitch(baseTexel.x, baseTexel.y, stride)];
  }

  if (a_sampler->m_filter != Sampler::Filter::MIN_MAG_MIP_LINEAR) {
    fprintf(stderr, "Unsupported filter is used.");
  }

  const int2 cornerTexel = make_int2(
    baseTexel.x < m_width  - 1 ? baseTexel.x + 1 : baseTexel.x,
    baseTexel.y < m_height - 1 ? baseTexel.y + 1 : baseTexel.y);

  const int offset0       = pitch(baseTexel.x  , baseTexel.y  , stride);
  const int offset1       = pitch(cornerTexel.x, baseTexel.y  , stride);
  const int offset2       = pitch(baseTexel.x  , cornerTexel.y, stride);
  const int offset3       = pitch(cornerTexel.x, cornerTexel.y, stride);

  const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
  const float4 line1Color = lerp(m_data[offset0], m_data[offset1], lerpCoefs.x);
  const float4 line2Color = lerp(m_data[offset2], m_data[offset3], lerpCoefs.x);

  return lerp(line1Color, line2Color, lerpCoefs.y);
}



template<typename DataType>
float2 Texture2D<DataType>::process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const
{ 
  float2 res = coord;

  switch (mode)
  {
    case Sampler::AddressMode::CLAMP: 
      //res = coord;            
      break;
    case Sampler::AddressMode::WRAP:  
      res = make_float2(coord.x - (int)(coord.x), coord.y - (int)(coord.y));      
      break;
    case Sampler::AddressMode::MIRROR:
    {
      const float u = static_cast<int>(coord.x) % 2 ? 1.0f - std::fmod(coord.x, 1.0f) : std::fmod(coord.x, 1.0f);
      const float v = static_cast<int>(coord.y) % 2 ? 1.0f - std::fmod(coord.y, 1.0f) : std::fmod(coord.y, 1.0f);
      res = make_float2(u, v);            
      break;
    }
    case Sampler::AddressMode::MIRROR_ONCE:
      res = make_float2(std::abs(coord.x), std::abs(coord.y));
      break;
    case Sampler::AddressMode::BORDER:
      *use_border_color = *use_border_color || coord.x < 0.0f || coord.x > 1.0f || coord.y < 0.0f || coord.y > 1.0f;
      break;      
    default:
      break;
  }

  return res = clamp(res, 0.0f, 1.0F);

  throw std::string("Uknown address mode.");
} 



float2 get_uv(const int x, const int y, const uint width, const uint height)
{
  const float u = (float)(x) / (float)(width);
  const float v = (float)(y) / (float)(height);
  return make_float2(u, v);
}
