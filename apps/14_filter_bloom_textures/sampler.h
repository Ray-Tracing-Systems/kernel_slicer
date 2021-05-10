#pragma once

#include "include/OpenCLMath.h"
#include <cstdint>
#include <cstdio>

struct Sampler {
  enum class AddressMode {
    WRAP,
    MIRROR,
    CLAMP,
    BORDER,
    MIRROR_ONCE
  };

  enum class Filter {
    MIN_MAG_MIP_POINT,
    MIN_MAG_POINT_MIP_LINEAR,
    MIN_POINT_MAG_LINEAR_MIP_POINT,
    MIN_POINT_MAG_MIP_LINEAR,
    MIN_LINEAR_MAG_MIP_POINT,
    MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    MIN_MAG_LINEAR_MIP_POINT,
    MIN_MAG_MIP_LINEAR,
    ANISOTROPIC,
    COMPARISON_MIN_MAG_MIP_POINT,
    COMPARISON_MIN_MAG_POINT_MIP_LINEAR,
    COMPARISON_MIN_POINT_MAG_LINEAR_MIP_POINT,
    COMPARISON_MIN_POINT_MAG_MIP_LINEAR,
    COMPARISON_MIN_LINEAR_MAG_MIP_POINT,
    COMPARISON_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    COMPARISON_MIN_MAG_LINEAR_MIP_POINT,
    COMPARISON_MIN_MAG_MIP_LINEAR,
    COMPARISON_ANISOTROPIC,
    MINIMUM_MIN_MAG_MIP_POINT,
    MINIMUM_MIN_MAG_POINT_MIP_LINEAR,
    MINIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT,
    MINIMUM_MIN_POINT_MAG_MIP_LINEAR,
    MINIMUM_MIN_LINEAR_MAG_MIP_POINT,
    MINIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    MINIMUM_MIN_MAG_LINEAR_MIP_POINT,
    MINIMUM_MIN_MAG_MIP_LINEAR,
    MINIMUM_ANISOTROPIC,
    MAXIMUM_MIN_MAG_MIP_POINT,
    MAXIMUM_MIN_MAG_POINT_MIP_LINEAR,
    MAXIMUM_MIN_POINT_MAG_LINEAR_MIP_POINT,
    MAXIMUM_MIN_POINT_MAG_MIP_LINEAR,
    MAXIMUM_MIN_LINEAR_MAG_MIP_POINT,
    MAXIMUM_MIN_LINEAR_MAG_POINT_MIP_LINEAR,
    MAXIMUM_MIN_MAG_LINEAR_MIP_POINT,
    MAXIMUM_MIN_MAG_MIP_LINEAR,
    MAXIMUM_ANISOTROPIC
  };

  enum class ComparisonFunc {
    NEVER,
    LESS,
    EQUAL,
    LESS_EQUAL,
    GREATER,
    NOT_EQUAL,
    GREATER_EQUAL,
    ALWAYS
  };
  //State structure from DX11
  // sampler state
  AddressMode m_addressU = AddressMode::WRAP;
  AddressMode m_addressV = AddressMode::WRAP;
  AddressMode m_addressW = AddressMode::WRAP;
  float4 m_borderColor = float4(0, 0, 0, 0);
  Filter m_filter = Filter::MIN_MAG_MIP_POINT;
  uint8_t m_maxAnisotropy = 1;
  float m_maxLOD = std::numeric_limits<float>::max();
  float m_minLOD = 0;
  float m_mipLODBias = 0;

  // sampler-comparison state
  ComparisonFunc m_comparisonFunc = ComparisonFunc::NEVER;
};

inline float process_coord(Sampler::AddressMode mode, float coord, bool &use_border_color) {
  switch (mode)
  {
    case Sampler::AddressMode::CLAMP:
      return clamp(coord, 0.f, 1.f);
    case Sampler::AddressMode::WRAP:
      return std::fmod(coord, 1.0);
    case Sampler::AddressMode::MIRROR:
      return static_cast<int>(coord) % 2 ? 1.0 - std::fmod(coord, 1.0) : std::fmod(coord, 1.0);
    case Sampler::AddressMode::MIRROR_ONCE:
      return clamp(std::abs(coord), 0.f, 1.f);
    case Sampler::AddressMode::BORDER:
      use_border_color = use_border_color || coord < 0.f || coord > 1.f;
      return coord;
  }
  throw std::string("Uknown address mode.");
}

inline float4 sample(Sampler sampler, __global const float4* data, int2 texSize, float2 uv) {
  bool useBorderColor = false;
  uv.x = process_coord(sampler.m_addressU, uv.x, useBorderColor);
  uv.y = process_coord(sampler.m_addressV, uv.y, useBorderColor);
  if (useBorderColor) {
    return sampler.m_borderColor;
  }

  const float2 textureSize(texSize.x, texSize.y);
  const float2 scaledUV = textureSize * uv;
  const int2 baseTexel = int2(scaledUV.x, scaledUV.y);
  const int stride = texSize.x;
  if (sampler.m_filter == Sampler::Filter::MIN_MAG_MIP_POINT) {
    return data[baseTexel.x + baseTexel.y * stride];
  }
  if (sampler.m_filter != Sampler::Filter::MIN_MAG_MIP_LINEAR) {
    fprintf(stderr, "Unsupported filter is used.");
  }
  const int2 cornerTexel = int2(
    baseTexel.x < texSize.x - 1 ? baseTexel.x + 1 : baseTexel.x,
    baseTexel.y < texSize.y - 1 ? baseTexel.y + 1 : baseTexel.y);

  const int offset0 = (baseTexel.x + baseTexel.y * stride);
  const int offset1 = (cornerTexel.x + baseTexel.y * stride);
  const int offset2 = (baseTexel.x + cornerTexel.y * stride);
  const int offset3 = (cornerTexel.x + cornerTexel.y * stride);

  const float2 lerpCoefs = scaledUV - float2(baseTexel.x, baseTexel.y);
  const float4 line1Color = lerp(data[offset0], data[offset1], lerpCoefs.x);
  const float4 line2Color = lerp(data[offset2], data[offset3], lerpCoefs.x);
  return lerp(line1Color, line2Color, lerpCoefs.y);
}
