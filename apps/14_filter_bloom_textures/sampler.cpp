#include "sampler.h"



float process_coord(const Sampler::AddressMode mode, const float coord, bool& use_border_color) 
{
  switch (mode)
  {
    case Sampler::AddressMode::CLAMP:
      return clamp(coord, 0.0f, 1.0f);
    case Sampler::AddressMode::WRAP:
      return std::fmod(coord, 1.0f);
    case Sampler::AddressMode::MIRROR:
      return static_cast<int>(coord) % 2 ? 1.0f - std::fmod(coord, 1.0f) : std::fmod(coord, 1.0f);
    case Sampler::AddressMode::MIRROR_ONCE:
      return clamp(std::abs(coord), 0.0f, 1.0f);
    case Sampler::AddressMode::BORDER:
      use_border_color = use_border_color || coord < 0.0f || coord > 1.0f;
      return coord;
    default:
      return coord;
  }
  throw std::string("Uknown address mode.");
}



float4 sample(Sampler sampler, __global const float4* data, int2 texSize, float2 uv) 
{
  bool useBorderColor = false;

  uv.x = process_coord(sampler.m_addressU, uv.x, useBorderColor);
  uv.y = process_coord(sampler.m_addressV, uv.y, useBorderColor);
  
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