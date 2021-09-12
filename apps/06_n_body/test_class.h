#pragma once

#include <vector>
#include <fstream>

#include "LiteMath.h"
using namespace LiteMath;

//enum class SIM_MODES
//{
//  GRAVITATIONAL,
//  ELECTROSTATIC
//};
//
//constexpr float getSimDT(SIM_MODES mode)
//{
//  switch(mode)
//  {
//    case SIM_MODES::ELECTROSTATIC:
//      return 1e-5f;
//    case SIM_MODES::GRAVITATIONAL:
//      return 5e-6f;
//    default:
//      return 1e-5f;
//  }
//}

class nBody
{
public:
  struct BodyState {
    float4 pos_weight;
    float4 vel_charge;
  };

  static constexpr int MODE = 1; //0 - gravitational, 1 - electrostatic
protected:
  static constexpr float dt = 1e-4f; //(MODE == 1) ? 1e-4f : 1e-5f;
  uint32_t m_seed;
  uint32_t m_iters;

  void kernel1D_GenerateBodies(uint32_t bodies_count);
  void kernel1D_UpdateVelocity(uint32_t bodies_count);
  void kernel1D_UpdatePosition(uint32_t bodies_count);
  void kernel1D_ReadData(BodyState *out_bodies, uint32_t bodies_count);
public:
  static constexpr bool PERIODIC_BOUNDARY_CONDITIONS = true;
  static constexpr float LATTICE_STEP = 5.f;
  static constexpr uint32_t LATTICE_RES = 10;
  static constexpr uint32_t BODIES_COUNT = 4096; //LATTICE_RES * LATTICE_RES * LATTICE_RES + 100;
  static constexpr float SOFTENING_CONST = 1e-5f;
  static constexpr float BOUNDARY = 60.0f;
  static constexpr float MASS = 5;
  static constexpr float CHARGE_MULT = 5e13;
  static constexpr float ELECTRON_CHARGE = 1.60218e-19;
  static constexpr float PERMETTIVITY = 8.85418782e-12;


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
