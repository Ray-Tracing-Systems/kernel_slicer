#pragma once

#include <vector>
#include <fstream>

#include "LiteMath.h"
using namespace LiteMath;

class nBody
{
public:
  struct BodyState {
    float4 pos_weight;
    float4 vel_charge;
  };
protected:
  static constexpr float dt = 5e-6f;
  uint32_t m_seed;
  uint32_t m_iters;

  void kernel1D_GenerateBodies(uint32_t bodies_count);
  void kernel1D_UpdateVelocity(uint32_t bodies_count);
  void kernel1D_UpdatePosition(uint32_t bodies_count);
  void kernel1D_ReadData(BodyState *out_bodies, uint32_t bodies_count);
public:
  static constexpr uint32_t BODIES_COUNT = 4096;
  static constexpr float SOFTENING_CONST = 1e-5f;
  static constexpr float MASS = 5;

  nBody() {
    m_bodies.resize(BODIES_COUNT);
  }
  void setParameters(uint32_t seed, uint32_t iters) {
    m_seed = seed;
    m_iters = iters;
  }
  void perform(BodyState *out_bodies);
  std::vector<BodyState> m_bodies;
};
