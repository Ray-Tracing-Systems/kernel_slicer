#include "include/OpenCLMath.h"
#include "test_class.h"

static float randFloat(const float min_value, const float max_value) {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * (max_value - min_value) + min_value;
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
  kernel1D_GenerateBodies(BODIES_COUNT);
  for (uint32_t i = 0; i < m_iters; ++i) {
    // dumpPositions();
    kernel1D_UpdateVelocity(BODIES_COUNT);
    kernel1D_UpdatePosition(BODIES_COUNT);
  }
  // dumpPositions();
}

void nBody::kernel1D_GenerateBodies(uint32_t bodies_count) {
  for (uint32_t i = 0; i < bodies_count; ++i) {
    m_bodies[i].pos_weight = randFloat4(float4(-1, -1, -1, -10), float4(1, 1, 1, 10));
    m_bodies[i].vel_charge = randFloat4(float4(-1, -1, -1, -1), float4(1, 1, 1, 1));
  }
}

float3 xyz(const float4& vec) {
  return float3(vec.x, vec.y, vec.z);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

static float pow3(float value) {
  return value * value * value;
}

void nBody::kernel1D_UpdateVelocity(uint32_t bodies_count) {
  for (uint32_t i = 0; i < bodies_count; ++i) {
    float3 acceleration(0, 0, 0);
    for (uint32_t j = 0; j < bodies_count; ++j) {
      if (i == j) {
        continue;
      }
      float3 bodyToBody = -xyz(m_bodies[j].pos_weight - m_bodies[i].pos_weight) * sgn(m_bodies[i].pos_weight.w);
      acceleration += bodyToBody * m_bodies[j].pos_weight.w / pow3(length(bodyToBody) + 1e-5f);
    }
    acceleration *= dt;
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
  }
}

// void nBody::dumpPositions() {
//   for (const auto& body: m_bodies) {
//     debugOutput.write(reinterpret_cast<const char*>(&body.pos_weight), sizeof(body.pos_weight));
//   }
// }

void n_body_cpu(uint32_t seed, uint32_t iterations) {
  nBody bodies(seed, iterations);
  bodies.perform();
}
