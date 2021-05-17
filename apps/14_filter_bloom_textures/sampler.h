#ifndef SAMPLER_H
#define SAMPLER_H

#include "include/OpenCLMath.h"
#include <iostream>

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
  AddressMode m_addressU      = AddressMode::WRAP;
  AddressMode m_addressV      = AddressMode::WRAP;
  AddressMode m_addressW      = AddressMode::WRAP;
  float4      m_borderColor   = float4(0.0f, 0.0f, 0.0f, 0.0f);
  Filter      m_filter        = Filter::MIN_MAG_MIP_POINT;
  uint8_t     m_maxAnisotropy = 1;
  float       m_maxLOD        = std::numeric_limits<float>::max();
  float       m_minLOD        = 0;
  float       m_mipLODBias    = 0;


  // sampler-comparison state
  ComparisonFunc m_comparisonFunc = ComparisonFunc::NEVER;
};


inline uint pitch(uint x, uint y, uint pitch) { return y * pitch + x; }  

#endif
