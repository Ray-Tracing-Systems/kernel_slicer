#pragma once

#include <array>
#include <fstream>

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
  void dumpPositions();
  std::ofstream debugOutput;
public:

  nBody(uint32_t seed, uint32_t iters) : m_seed(seed), m_iters(iters) {
    debugOutput = std::ofstream("BodiesDumpCPU.bin", std::ios::binary);
    debugOutput.write(reinterpret_cast<const char*>(&BODIES_COUNT), sizeof(BODIES_COUNT));
    debugOutput.write(reinterpret_cast<const char*>(&m_iters), sizeof(m_iters));
  }
  void perform();
  std::array<BodyState, BODIES_COUNT> m_bodies;
};
