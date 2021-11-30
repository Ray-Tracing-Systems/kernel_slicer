#include "test_class.h"
#include "include/crandom.h"

static uint32_t nextRandValue(const uint32_t value) {
  return value * 22695477 + 1; // Borland C random
}

static float4 randFloat4(float4 min_value, float4 max_value, const uint32_t threadId) {
  RandomGen gen = RandomGenInit(nextRandValue(threadId));
  return rndFloat4_Pseudo(&gen)*(max_value - min_value) + min_value;
}

void nBody::perform(BodyState *out_bodies) {
  kernel1D_GenerateBodies(BODIES_COUNT);
  for (uint32_t i = 0; i < m_iters; ++i) {
    kernel1D_UpdateVelocity(BODIES_COUNT);
    kernel1D_UpdatePosition(BODIES_COUNT);
  }
  memcpy(out_bodies, m_bodies.data(), m_bodies.size()*sizeof(BodyState));
}

void nBody::kernel1D_GenerateBodies(uint32_t bodies_count) {
  for (uint32_t i = 0; i < bodies_count; ++i) {
    m_bodies[i].pos_weight = randFloat4(make_float4(-BOUNDARY * 0.1f, -BOUNDARY* 0.1f, -BOUNDARY* 0.1f, MASS),
                                        make_float4(BOUNDARY* 0.1f, BOUNDARY* 0.1f, BOUNDARY* 0.1f, MASS), i);
//    m_bodies[i].vel_charge = randFloat4(make_float4(-1, -1, -1, -ELECTRON_CHARGE * CHARGE_MULT),
//                                        make_float4(1, 1, 1, -ELECTRON_CHARGE * CHARGE_MULT), i*i + i*7 + 1);


    if(i % 2 == 0)
      m_bodies[i].vel_charge = make_float4(0, 0, 0, ELECTRON_CHARGE * CHARGE_MULT);
    else
      m_bodies[i].vel_charge = make_float4(0, 0, 0, -ELECTRON_CHARGE * CHARGE_MULT);
  }
}

//void nBody::kernel1D_GenerateBodies(uint32_t bodies_count)
//{
//
//  for (uint32_t i = 0; i < bodies_count; ++i) {
//    if(i >= LATTICE_RES * LATTICE_RES * LATTICE_RES)
//    {
//      m_bodies[i].pos_weight = randFloat4(make_float4(-BOUNDARY * 0.25f, -BOUNDARY* 0.25f, -BOUNDARY* 0.25f, MASS),
//                                          make_float4(BOUNDARY* 0.25f, BOUNDARY* 0.25f, BOUNDARY* 0.25f, MASS), i);
//      m_bodies[i].vel_charge = randFloat4(make_float4(-1, -1, -1, -ELECTRON_CHARGE * CHARGE_MULT),
//                                          make_float4(1, 1, 1, -ELECTRON_CHARGE * CHARGE_MULT), i*i + i*7 + 1);
//      m_bodies[i].vel_charge.w = -ELECTRON_CHARGE * CHARGE_MULT;
//    }
//    else
//    {
//      float3 coord = make_float3(-BOUNDARY* 0.25f, -BOUNDARY* 0.25f, -BOUNDARY* 0.25f);
//      uint32_t x = i % LATTICE_RES;
//      uint32_t y = i / LATTICE_RES / LATTICE_RES;
//      uint32_t z = (i / LATTICE_RES) % LATTICE_RES;
//      //    m_bodies[i].pos_weight = randFloat4(make_float4(-1, -1, -1, MASS), make_float4(1, 1, 1, MASS), i);
//      m_bodies[i].pos_weight = make_float4(coord.x + x * LATTICE_STEP,
//                                           coord.y + y * LATTICE_STEP,
//                                           coord.z + z * LATTICE_STEP,
//                                           MASS * 10 * 1e4);
//      m_bodies[i].vel_charge = make_float4(0, 0, 0, ELECTRON_CHARGE * CHARGE_MULT);
//    }
//  }
//}

static float3 xyz(const float4 vec) {
  return make_float3(vec.x, vec.y, vec.z);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static float pow3(float value) {
  return value * value * value;
}

void nBody::kernel1D_UpdateVelocity(uint32_t bodies_count) {
  for (uint32_t i = 0; i < bodies_count; ++i) {
    float3 acceleration = make_float3(0, 0, 0);
    for (uint32_t j = 0; j < m_bodies.size(); ++j) {
//      if(m_bodies[i].pos_weight.w > MASS)
//        continue;
      if (i == j) {
        continue;
      }
      float3 distance = xyz(m_bodies[j].pos_weight - m_bodies[i].pos_weight); // * sgn(m_bodies[i].pos_weight.w);
      float distSqr = dot(distance, distance) + SOFTENING_CONST;
      float invDistCube = 1.0f/std::sqrt(pow3(distSqr));
      float3 gravitational = distance * m_bodies[j].pos_weight.w * invDistCube;

      float coeff = m_bodies[i].vel_charge.w / (4 * M_PI * PERMETTIVITY);
      float3 electrostatic = coeff * m_bodies[j].vel_charge.w * (+1.f * distance / distSqr);

      if(MODE == 1)
      {
        acceleration += electrostatic;
      }
      else if(MODE == 0)
      {
        acceleration += gravitational;
      }
      else
        acceleration += gravitational + electrostatic;
    }
    acceleration *= dt;
    acceleration /= m_bodies[i].pos_weight.w;
    m_bodies[i].vel_charge.x += acceleration.x;
    m_bodies[i].vel_charge.y += acceleration.y;
    m_bodies[i].vel_charge.z += acceleration.z;
  }
}

void nBody::kernel1D_UpdatePosition(uint32_t bodies_count) {
  for (uint32_t i = 0; i < bodies_count; ++i) {
    m_bodies[i].pos_weight.x += m_bodies[i].vel_charge.x * dt;
    m_bodies[i].pos_weight.y += m_bodies[i].vel_charge.y * dt;
    m_bodies[i].pos_weight.z += m_bodies[i].vel_charge.z * dt;

    if(PERIODIC_BOUNDARY_CONDITIONS)
    {
      m_bodies[i].pos_weight.x = m_bodies[i].pos_weight.x > BOUNDARY ? m_bodies[i].pos_weight.x - 2 * BOUNDARY : m_bodies[i].pos_weight.x;
      m_bodies[i].pos_weight.y = m_bodies[i].pos_weight.y > BOUNDARY ? m_bodies[i].pos_weight.y - 2 * BOUNDARY : m_bodies[i].pos_weight.y;
      m_bodies[i].pos_weight.z = m_bodies[i].pos_weight.z > BOUNDARY ? m_bodies[i].pos_weight.z - 2 * BOUNDARY : m_bodies[i].pos_weight.z;

      m_bodies[i].pos_weight.x = m_bodies[i].pos_weight.x < -BOUNDARY ? 2 * BOUNDARY + m_bodies[i].pos_weight.x : m_bodies[i].pos_weight.x;
      m_bodies[i].pos_weight.y = m_bodies[i].pos_weight.y < -BOUNDARY ? 2 * BOUNDARY + m_bodies[i].pos_weight.y : m_bodies[i].pos_weight.y;
      m_bodies[i].pos_weight.z = m_bodies[i].pos_weight.z < -BOUNDARY ? 2 * BOUNDARY + m_bodies[i].pos_weight.z : m_bodies[i].pos_weight.z;
    }
  }
}

void nBody::kernel1D_ReadData(BodyState *out_bodies, uint32_t bodies_count) {
  for (uint32_t i = 0; i < bodies_count; ++i) {
    out_bodies[i] = m_bodies[i];
  }
}

std::vector<nBody::BodyState> n_body_cpu(uint32_t seed, uint32_t iterations) {
  nBody bodies;
  std::vector<nBody::BodyState> outBodies(nBody::BODIES_COUNT);
  bodies.setParameters(seed, iterations);
  bodies.perform(outBodies.data());
  return outBodies;
}
