#include <iostream>
#include <test_class.h>
#include "render/points_render.h"
#include "glfw_window.h"
#include "ArgParser.h"

std::vector<nBody::BodyState> n_body_cpu(uint32_t seed, uint32_t iterations);
std::vector<nBody::BodyState> n_body_gpu(uint32_t seed, uint32_t iterations);

const float EPS = 1e-3f;

inline float errFloat4(const float4& a, const float4& b) 
{
  return std::max(std::max(std::abs(a.x - b.x), std::abs(a.y - b.y)), 
                  std::max(std::abs(a.z - b.z), std::abs(a.w - b.w)));
}

int compute_main()
{
  const uint32_t SEED       = 42;
  const uint32_t ITERATIONS = 10;
  auto out_cpu = n_body_cpu(SEED, ITERATIONS);
  auto out_gpu = n_body_gpu(SEED, ITERATIONS);
  bool failed = false;

  std::vector<uint32_t> badId; badId.reserve(10);
  std::vector<uint32_t> badId2; badId2.reserve(10);

  float maxErr   = 0.0f;
  float avgError = 0.0f;

  for (uint32_t i = 0; i < out_cpu.size(); ++i)
  {
    float errPos = errFloat4(out_cpu[i].pos_weight, out_gpu[i].pos_weight);
    float errVel = errFloat4(out_cpu[i].vel_charge, out_gpu[i].vel_charge);

    if (errPos > EPS)
    {
      if(badId.size() + 1 < badId.capacity())
        badId.push_back(i);
    }

    if (errVel > EPS)
    {
      if(badId2.size() + 1 < badId2.capacity())
        badId2.push_back(i);
    }

    avgError += std::max(errVel,errPos);
    maxErr = std::max(maxErr, std::max(errVel,errPos));
  }

  avgError /= float(out_cpu.size());

  std::cout << "maxErr = " << maxErr << std::endl;
  std::cout << "avgErr = " << avgError << std::endl;

  for(const auto i : badId)
  {
    failed = true;
    std::cout << "Wrong position " << i << std::endl;
    std::cout << "CPU value: " << out_cpu[i].pos_weight.x << "\t" << out_cpu[i].pos_weight.y << "\t" << out_cpu[i].pos_weight.z << "\t" << out_cpu[i].pos_weight.w << std::endl;
    std::cout << "GPU value: " << out_gpu[i].pos_weight.x << "\t" << out_gpu[i].pos_weight.y << "\t" << out_gpu[i].pos_weight.z << "\t" << out_gpu[i].pos_weight.w << std::endl;
  }

  for(const auto i : badId2)
  {
    failed = true;
    std::cout << "Wrong velocity " << i << std::endl;
    std::cout << "CPU value: " << out_cpu[i].vel_charge.x << "\t" << out_cpu[i].vel_charge.y << "\t" << out_cpu[i].vel_charge.z << "\t" << out_cpu[i].vel_charge.w << std::endl;
    std::cout << "GPU value: " << out_gpu[i].vel_charge.x << "\t" << out_gpu[i].vel_charge.y << "\t" << out_gpu[i].vel_charge.z << "\t" << out_gpu[i].vel_charge.w << std::endl;
  }

  if (failed) {
    std::cout << "FAIL" << std::endl;
    return -1;
  } else {
    std::cout << "OK" << std::endl;
  }
  return 0;
}

int graphics_main()
{
  std::shared_ptr<IRender> app = CreateRender(1024, 1024, RenderEngineType::POINTS_RENDER);
  if(app == nullptr)
  {
    std::cout << "Can't create render of specified type\n";
    return 1;
  }
  auto* window = Init(app, 0);

  app->LoadScene("", true);

  MainLoop(app, window);
  return 0;
}


int main(int argc, const char** argv)
{
  ArgParser args(argc, argv);
  bool runCmdLineMode = args.hasOption("--test");

  if(runCmdLineMode)
    return compute_main();
  else
    graphics_main();

  return 0;
}
