#include "sampler.h"



float2 process_coord(const Sampler::AddressMode mode, const float2 coord, bool& use_border_color) 
{
  switch (mode)
  {
    case Sampler::AddressMode::CLAMP: 
      return make_float2(    clamp(coord.x, 0.0f, 1.0f), clamp(coord.y, 0.0f, 1.0f));            
    case Sampler::AddressMode::WRAP:  
      return make_float2(std::fmod(coord.x, 1.0f),   std::fmod(coord.y, 1.0f));      
    case Sampler::AddressMode::MIRROR:
    {
      const float u = static_cast<int>(coord.x) % 2 ? 1.0f - std::fmod(coord.x, 1.0f) : std::fmod(coord.x, 1.0f);
      const float v = static_cast<int>(coord.y) % 2 ? 1.0f - std::fmod(coord.y, 1.0f) : std::fmod(coord.y, 1.0f);
      return make_float2(u, v);            
    }
    case Sampler::AddressMode::MIRROR_ONCE:
      return make_float2(clamp(std::abs(coord.x), 0.0f, 1.0f), clamp(std::abs(coord.y), 0.0f, 1.0f));
    case Sampler::AddressMode::BORDER:
      use_border_color = use_border_color || coord.x < 0.0f || coord.x > 1.0f || coord.y < 0.0f || coord.y > 1.0f;
      return coord;
    default:
      return coord;
  }
  throw std::string("Uknown address mode.");
}



float4 sample(Sampler sampler, __global const float4* data, int2 texSize, float2 uv) 
{
  bool useBorderColor = false;

  uv = process_coord(sampler.m_addressU, uv, useBorderColor);
    
  if (useBorderColor) {
    return sampler.m_borderColor;
  }

  const float2 textureSize(texSize.x, texSize.y);
  const float2 scaledUV = textureSize * uv;
  const int2 baseTexel  = int2(scaledUV.x, scaledUV.y);
  const int stride      = texSize.x;

  if (sampler.m_filter == Sampler::Filter::MIN_MAG_MIP_POINT) {
    return data[baseTexel.y * stride + baseTexel.x];
  }

  if (sampler.m_filter != Sampler::Filter::MIN_MAG_MIP_LINEAR) {
    fprintf(stderr, "Unsupported filter is used.");
  }

  const int2 cornerTexel = int2(
    baseTexel.x < texSize.x - 1 ? baseTexel.x + 1 : baseTexel.x,
    baseTexel.y < texSize.y - 1 ? baseTexel.y + 1 : baseTexel.y);

  const int offset0       = (baseTexel.y   * stride + baseTexel.x  );
  const int offset1       = (baseTexel.y   * stride + cornerTexel.x);
  const int offset2       = (cornerTexel.y * stride + baseTexel.x  );
  const int offset3       = (cornerTexel.y * stride + cornerTexel.x);

  const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
  const float4 line1Color = lerp(data[offset0], data[offset1], lerpCoefs.x);
  const float4 line2Color = lerp(data[offset2], data[offset3], lerpCoefs.x);

  return lerp(line1Color, line2Color, lerpCoefs.y);
}