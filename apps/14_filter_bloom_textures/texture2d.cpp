#include "texture2d.h"

///////////////////////////////////////////////////////////////////////

template<> float4 Texture2D<float4>::sample(const Sampler& a_sampler, float2 a_uv) const
{
  bool useBorderColor = false;

  a_uv = process_coord(a_sampler.addressU, a_uv, &useBorderColor);
    
  if (useBorderColor) {
    return a_sampler.borderColor;
  }

  const float2 textureSize = make_float2(m_width, m_height);
  const float2 scaledUV    = textureSize * a_uv;
  const int2   baseTexel   = int2(scaledUV.x, scaledUV.y);
  const int    stride      = m_width;

  switch (a_sampler.filter)
  {
  case Sampler::Filter::MIN_MAG_MIP_POINT:
    return m_data[pitch(baseTexel.x, baseTexel.y, stride)];
  case Sampler::Filter::MIN_MAG_LINEAR_MIP_POINT:
  {
    const int2 cornerTexel  = int2(baseTexel.x < m_width  - 1 ? baseTexel.x + 1 : baseTexel.x,
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
  default:
    fprintf(stderr, "Unsupported filter is used.");
    break;
  }

  return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
}

template<> float4 Texture2D<uchar4>::sample(const Sampler& a_sampler, float2 a_uv) const
{
  bool useBorderColor = false;

  a_uv = process_coord(a_sampler.addressU, a_uv, &useBorderColor);
    
  if (useBorderColor) {
    return a_sampler.borderColor;
  }

  const float2 textureSize = make_float2(m_width, m_height);
  const float2 scaledUV    = textureSize * a_uv;
  const int2   baseTexel   = make_int2(scaledUV.x, scaledUV.y);
  const int    stride      = m_width;

  switch (a_sampler.filter)
  {
  case Sampler::Filter::MIN_MAG_MIP_POINT:
  {
    const uchar4 uData = m_data[pitch(baseTexel.x, baseTexel.y, stride)];
    return (1.0f/255.0f)*float4(uData.x, uData.y, uData.z, uData.w);
  }
  case Sampler::Filter::MIN_MAG_LINEAR_MIP_POINT:
  {
    const int2 cornerTexel  = make_int2(baseTexel.x < m_width  - 1 ? baseTexel.x + 1 : baseTexel.x,
                                        baseTexel.y < m_height - 1 ? baseTexel.y + 1 : baseTexel.y);

    const int offset0       = pitch(baseTexel.x  , baseTexel.y  , stride);
    const int offset1       = pitch(cornerTexel.x, baseTexel.y  , stride);
    const int offset2       = pitch(baseTexel.x  , cornerTexel.y, stride);
    const int offset3       = pitch(cornerTexel.x, cornerTexel.y, stride);

    const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
    const uchar4 uData1     = lerp(m_data[offset0], m_data[offset1], lerpCoefs.x);
    const uchar4 uData2     = lerp(m_data[offset2], m_data[offset3], lerpCoefs.x);
    const float4 line1Color = (1.0f/255.0f)*float4(uData1.x, uData1.y, uData1.z, uData1.w);
    const float4 line2Color = (1.0f/255.0f)*float4(uData2.x, uData2.y, uData2.z, uData2.w);

    return lerp(line1Color, line2Color, lerpCoefs.y);
  }     
  default:
    fprintf(stderr, "Unsupported filter is used.");
    break;
  }

  return make_float4(0.0F, 0.0F, 0.0F, 0.0F);
}



template<typename Type>
float2 Texture2D<Type>::process_coord(const Sampler::AddressMode mode, const float2 coord, bool* use_border_color) const
{ 
  float2 res = coord;

  switch (mode)
  {
    case Sampler::AddressMode::CLAMP: 
      //res = coord;            
      break;
    case Sampler::AddressMode::WRAP:  
    {
      res = make_float2(std::fmod(coord.x, 1.0), std::fmod(coord.y, 1.0));            
      if (res.x < 0.0F) res.x += 1.0F;
      if (res.y < 0.0F) res.y += 1.0F;      
      break;
    }
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

  return clamp(res, 0.0f, 1.0F);
} 
