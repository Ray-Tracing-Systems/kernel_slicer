#ifndef SAMPLER_H
#define SAMPLER_H

#include "LiteMath.h"
using namespace LiteMath;

#include <iostream>

struct Sampler {

  enum class AddressMode {
    WRAP        = 0,
    MIRROR      = 1,
    CLAMP       = 2,
    BORDER      = 3,
    MIRROR_ONCE = 4,
  };

  enum class Filter {
    NEAREST = 0,
    LINEAR  = 1,
  };

  // sampler state
  //
  AddressMode addressU      = AddressMode::WRAP;
  AddressMode addressV      = AddressMode::WRAP;
  AddressMode addressW      = AddressMode::WRAP;
  float4      borderColor   = float4(0.0f, 0.0f, 0.0f, 0.0f);
  Filter      filter        = Filter::NEAREST;
  uint32_t    maxAnisotropy = 1;
  uint32_t    maxLOD        = 32;
  uint32_t    minLOD        = 0;
  uint32_t    mipLODBias    = 0;

};

#endif
