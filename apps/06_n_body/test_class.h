#pragma once

#include <vector>
#include <fstream>

#include "include/OpenCLMath.h"

class nBody
{
protected:
  static constexpr float dt = 1e-3f;
  struct BodyState {
    float4 pos_weight;
    float4 vel_charge;
  };
  uint32_t m_seed;
  uint32_t m_iters;

  void kernel1D_GenerateBodies(uint32_t bodies_count);
  void kernel1D_UpdateVelocity(uint32_t bodies_count);
  void kernel1D_UpdatePosition(uint32_t bodies_count);
  // DEBUG
  void dumpPositions();
  std::ofstream debugOutput;
public:
  static constexpr uint32_t BODIES_COUNT = 500;

  nBody() {
    m_bodies.resize(BODIES_COUNT);
  }
  void setParameters(uint32_t seed, uint32_t iters) {
    m_seed = seed;
    m_iters = iters;
    debugOutput = std::ofstream("BodiesDumpCPU.bin", std::ios::binary);
    debugOutput.write(reinterpret_cast<const char*>(&BODIES_COUNT), sizeof(BODIES_COUNT));
    debugOutput.write(reinterpret_cast<const char*>(&m_iters), sizeof(m_iters));
  }
  void perform();
  std::vector<BodyState> m_bodies;
};
