#include "test_class.h"
#include "include/crandom.h"
#include <chrono>
#include <iostream>

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
    m_bodies[i].pos_weight = randFloat4(make_float4(-1, -1, -1, 10), make_float4(1, 1, 1, 10), i);
    m_bodies[i].vel_charge = randFloat4(make_float4(-1, -1, -1, -1), make_float4(1, 1, 1, 1), i*i + i*7 + 1);
  }
}

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

  #pragma omp parallel for
  for (uint32_t i = 0; i < bodies_count; ++i) {
    float3 acceleration = make_float3(0, 0, 0);
    for (uint32_t j = 0; j < m_bodies.size(); ++j) {
      if (i == j) {
        continue;
      }
      float3 bodyToBody = xyz(m_bodies[j].pos_weight - m_bodies[i].pos_weight); // * sgn(m_bodies[i].pos_weight.w);
      acceleration += bodyToBody * m_bodies[j].pos_weight.w / (pow3(length(bodyToBody)) + 1e-5f);
    }
    acceleration *= dt;
    m_bodies[i].vel_charge.x += acceleration.x;
    m_bodies[i].vel_charge.y += acceleration.y;
    m_bodies[i].vel_charge.z += acceleration.z;
  }
}

void nBody::kernel1D_UpdatePosition(uint32_t bodies_count) {
  #pragma omp parallel for
  for (uint32_t i = 0; i < bodies_count; ++i) {
    m_bodies[i].pos_weight.x += m_bodies[i].vel_charge.x * dt;
    m_bodies[i].pos_weight.y += m_bodies[i].vel_charge.y * dt;
    m_bodies[i].pos_weight.z += m_bodies[i].vel_charge.z * dt;
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

  for(int i=0;i<10;i++)
  {
    auto start = std::chrono::high_resolution_clock::now();
    bodies.perform(outBodies.data());

    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for 'bodies.perform' " << std::endl;
  }

  return outBodies;
}
