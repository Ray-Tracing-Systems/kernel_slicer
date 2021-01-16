#include <iostream>
#include <test_class.h>

std::vector<nBody::BodyState> n_body_cpu(uint32_t seed, uint32_t iterations);
std::vector<nBody::BodyState> n_body_gpu(uint32_t seed, uint32_t iterations);

bool compareFloat4(const float4& a, const float4& b) {
  const float EPS = 1e-6f;
  return std::abs(a.x - b.x) > EPS || std::abs(a.y - b.y) > EPS || std::abs(a.z - b.z) > EPS || std::abs(a.w - b.w) > EPS;
}

int main(int argc, const char** argv)
{
  const uint32_t SEED = 42;
  const uint32_t ITERATIONS = 1;
  auto out_cpu = n_body_cpu(SEED, ITERATIONS);
  auto out_gpu = n_body_gpu(SEED, ITERATIONS);
  bool failed = false;
  for (uint32_t i = 0; i < out_cpu.size(); ++i) {
    if (compareFloat4(out_cpu[i].pos_weight, out_gpu[i].pos_weight)) {
      std::cout << "Wrong position " << i << std::endl;
      std::cout << "CPU value: " << out_cpu[i].pos_weight.x << " " << out_cpu[i].pos_weight.y << " " << out_cpu[i].pos_weight.z << " " << out_cpu[i].pos_weight.w << std::endl;
      std::cout << "GPU value: " << out_gpu[i].pos_weight.x << " " << out_gpu[i].pos_weight.y << " " << out_gpu[i].pos_weight.z << " " << out_gpu[i].pos_weight.w << std::endl;
      failed = true;
      break;
    }
    if (compareFloat4(out_cpu[i].vel_charge, out_gpu[i].vel_charge)) {
      std::cout << "Wrong velocity " << i << std::endl;
      std::cout << "CPU value: " << out_cpu[i].vel_charge.x << " " << out_cpu[i].vel_charge.y << " " << out_cpu[i].vel_charge.z << " " << out_cpu[i].vel_charge.w << std::endl;
      std::cout << "GPU value: " << out_gpu[i].vel_charge.x << " " << out_gpu[i].vel_charge.y << " " << out_gpu[i].vel_charge.z << " " << out_gpu[i].vel_charge.w << std::endl;
      failed = true;
      break;
    }
  }
  if (failed) {
    std::cout << "FAIL" << std::endl;
  } else {
    std::cout << "OK" << std::endl;
  }
  return 0;
}
