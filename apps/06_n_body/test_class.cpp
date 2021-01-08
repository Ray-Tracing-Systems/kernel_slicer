#include <iostream>
#include <fstream>
#include <sstream>

#include "include/LiteMath.h"

#include "test_class.h"

using namespace LiteMath;

static float randFloat(const float min_value, const float max_value) {
  return static_cast<float>(rand()) / RAND_MAX * (max_value - min_value) + min_value;
}

static float4 randFloat4(const float4& min_value, const float4& max_value) {
  return float4(
    randFloat(min_value.x, max_value.x),
    randFloat(min_value.y, max_value.y),
    randFloat(min_value.z, max_value.z),
    randFloat(min_value.w, max_value.w)
  );
}


void nBody::perform() {
  kernel1D_GenerateBodies();
  for (uint32_t i = 0; i < m_iters; ++i) {
    dumpPositions(i);
    kernel1D_UpdateVelocity();
    kernel1D_UpdatePosition();
  }
  dumpPositions(m_iters);
}

void nBody::kernel1D_GenerateBodies() {
  for (auto& body : m_bodies) {
    body.pos_weight = randFloat4(float4(-1, -1, -1, 0), float4(1, 1, 1, 10));
    body.vel_charge = randFloat4(float4(-1, -1, -1, -1), float4(1, 1, 1, 1));
  }
}

float3 xyz(const float4& vec) {
  return float3(vec.x, vec.y, vec.z);
}

void nBody::kernel1D_UpdateVelocity() {
  for (uint32_t i = 0; i < m_bodies.size(); ++i) {
    float3 acceleration(0, 0, 0);
    for (uint32_t j = 0; j < m_bodies.size(); ++j) {
      if (i == j) {
        continue;
      }
      float3 bodyToBody = xyz(m_bodies[j].pos_weight - m_bodies[i].pos_weight);
      acceleration += bodyToBody * m_bodies[j].pos_weight.w / std::pow(length(bodyToBody) + 1e-5f, 3.0f);
    }
    acceleration *= dt;
    m_bodies[i].vel_charge.x += acceleration.x;
    m_bodies[i].vel_charge.y += acceleration.y;
    m_bodies[i].vel_charge.z += acceleration.z;
  }
}

void nBody::kernel1D_UpdatePosition() {
  for (auto& body: m_bodies) {
    body.pos_weight.x += body.vel_charge.x * dt;
    body.pos_weight.y += body.vel_charge.y * dt;
    body.pos_weight.z += body.vel_charge.z * dt;
  }
}

void nBody::dumpPositions(uint32_t marker) {
  std::stringstream ss;
  ss << "dumps/BodiesDumpCPU_" << marker;
  std::ofstream output(ss.str(), std::ios::binary);
  output.write(reinterpret_cast<const char*>(&BODIES_COUNT), sizeof(BODIES_COUNT));
  for (const auto& body: m_bodies) {
    output.write(reinterpret_cast<const char*>(&body.pos_weight), sizeof(body.pos_weight));
  }
  output.close();
}

void n_body_cpu(uint32_t seed, uint32_t iterations) {
  nBody bodies(seed, iterations);
  bodies.perform();
}
