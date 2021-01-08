#pragma once

#include <array>

class nBody
{
  static constexpr uint32_t BODIES_COUNT = 500;
  static constexpr float dt = 1e-3f;
  struct BodyState {
    LiteMath::float4 pos_weight;
    LiteMath::float4 vel_charge;
  };
  uint32_t m_seed;
  uint32_t m_iters;

  void kernel1D_GenerateBodies();
  void kernel1D_UpdateVelocity();
  void kernel1D_UpdatePosition();
  ///DEBUG
  void dumpPositions(uint32_t marker);
public:

  nBody(uint32_t seed, uint32_t iters) : m_seed(seed), m_iters(iters) {}
  void perform();
  std::array<BodyState, BODIES_COUNT> m_bodies;
};
