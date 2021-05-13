#include "texture2d.h"


template<> float4 Texture2D<float4>::sample(const Sampler* a_sampler, float2 a_uv, const uint a_width, const uint a_height) const
{
  bool useBorderColor = false;

  a_uv = process_coord(a_sampler->m_addressU, a_uv, &useBorderColor);
    
  if (useBorderColor) {
    return a_sampler->m_borderColor;
  }

  const float2 textureSize = make_float2(a_width, a_height);
  const float2 scaledUV    = textureSize * make_float2(a_uv.x, a_uv.y);
  const int2   baseTexel   = make_int2(scaledUV.x, scaledUV.y);
  const int    stride      = a_width;

  if (a_sampler->m_filter == Sampler::Filter::MIN_MAG_MIP_POINT) {
    return m_data[baseTexel.y * stride + baseTexel.x];
  }

  if (a_sampler->m_filter != Sampler::Filter::MIN_MAG_MIP_LINEAR) {
    fprintf(stderr, "Unsupported filter is used.");
  }

  const int2 cornerTexel = make_int2(
    baseTexel.x < a_width  - 1 ? baseTexel.x + 1 : baseTexel.x,
    baseTexel.y < a_height - 1 ? baseTexel.y + 1 : baseTexel.y);

  const int offset0       = (baseTexel.y   * stride + baseTexel.x  );
  const int offset1       = (baseTexel.y   * stride + cornerTexel.x);
  const int offset2       = (cornerTexel.y * stride + baseTexel.x  );
  const int offset3       = (cornerTexel.y * stride + cornerTexel.x);

  const float2 lerpCoefs  = scaledUV - float2(baseTexel.x, baseTexel.y);
  const float4 line1Color = lerp(m_data[offset0], m_data[offset1], lerpCoefs.x);
  const float4 line2Color = lerp(m_data[offset2], m_data[offset3], lerpCoefs.x);

  return lerp(line1Color, line2Color, lerpCoefs.y);
}



float2 get_uv(const int x, const int y, const uint width, const uint height)
{
  const float u = (float)(x) / (float)(width);
  const float v = (float)(y) / (float)(height);
  return make_float2(u, v);
}
