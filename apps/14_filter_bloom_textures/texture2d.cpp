#include "texture2d.h"


template<> float4 Texture2D<float4>::sample(const Sampler* sampler, float2 uv, const int2 texSize) const
{
  bool useBorderColor = false;

  uv.x = process_coord(sampler->m_addressU, uv.x, useBorderColor);
  uv.y = process_coord(sampler->m_addressV, uv.y, useBorderColor);
  
  if (useBorderColor) {
    return sampler->m_borderColor;
  }

  const float2 textureSize = make_float2(texSize.x, texSize.y);
  const float2 scaledUV    = textureSize * uv;
  const int2   baseTexel   = make_int2(scaledUV.x, scaledUV.y);
  const int    stride      = texSize.x;

  if (sampler->m_filter == Sampler::Filter::MIN_MAG_MIP_POINT) {
    return read_pixel(baseTexel.y * stride + baseTexel.x);
  }

  if (sampler->m_filter != Sampler::Filter::MIN_MAG_MIP_LINEAR) {
    fprintf(stderr, "Unsupported filter is used.");
  }

  const int2 cornerTexel = make_int2(
    baseTexel.x < texSize.x - 1 ? baseTexel.x + 1 : baseTexel.x,
    baseTexel.y < texSize.y - 1 ? baseTexel.y + 1 : baseTexel.y);

  const int offset0       = (baseTexel.y   * stride + baseTexel.x  );
  const int offset1       = (baseTexel.y   * stride + cornerTexel.x);
  const int offset2       = (cornerTexel.y * stride + baseTexel.x  );
  const int offset3       = (cornerTexel.y * stride + cornerTexel.x);

  const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
  const float4 line1Color = lerp(read_pixel(offset0), read_pixel(offset1), lerpCoefs.x);
  const float4 line2Color = lerp(read_pixel(offset2), read_pixel(offset3), lerpCoefs.x);

  return lerp(line1Color, line2Color, lerpCoefs.y);
}