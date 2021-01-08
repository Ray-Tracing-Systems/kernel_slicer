#include <iostream>

void n_body_cpu(uint32_t seed, uint32_t iterations);
void n_body_gpu(uint32_t seed, uint32_t iterations);

int main(int argc, const char** argv)
{
  const uint32_t SEED = 42;
  const uint32_t ITERATIONS = 200;
  n_body_cpu(SEED, ITERATIONS);
  return 0;
}
