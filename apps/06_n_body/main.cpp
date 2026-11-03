#include <iostream>
#include <test_class.h>
#include "render/points_render.h"
#include "glfw_window.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

std::vector<nBody::BodyState> n_body_cpu(uint32_t seed, uint32_t iterations);
std::vector<nBody::BodyState> n_body_gpu(uint32_t seed, uint32_t iterations);
#ifdef USE_ISPC
std::vector<nBody::BodyState> n_body_ispc(uint32_t seed, uint32_t iterations);
#endif

// summary values of the simulation result, they are compared between CPU and GPU/ISPC runs
void save_summary(const std::vector<nBody::BodyState>& a_bodies, const char* a_fileName)
{
  double3 sumPos(0.0);
  double  sumDist = 0.0, sumSpeed = 0.0;
  float   maxSpeed = 0.0f;
  for(const auto& body : a_bodies)
  {
    const float3 pos = to_float3(body.pos_weight);
    const float3 vel = to_float3(body.vel_charge);
    sumPos   += double3(pos.x, pos.y, pos.z);
    sumDist  += double(length(pos));
    sumSpeed += double(length(vel));
    maxSpeed  = std::max(maxSpeed, length(vel));
  }

  const double invCount = 1.0/double(a_bodies.size());
  float avgPos[3] = {float(sumPos.x*invCount), float(sumPos.y*invCount), float(sumPos.z*invCount)};

  JSONLog::write("avg position", avgPos);
  JSONLog::write("avg distance", float(sumDist*invCount));
  JSONLog::write("avg speed",    float(sumSpeed*invCount));
  JSONLog::write("max speed",    maxSpeed);
  JSONLog::saveToFile(a_fileName);
}

int compute_main(bool a_onGPU, bool a_isISPC)
{
  const uint32_t SEED       = 42;
  const uint32_t ITERATIONS = 10;

  if(a_isISPC)
  {
    #ifdef USE_ISPC
    save_summary(n_body_ispc(SEED, ITERATIONS), "zout_ispc.json");
    #else
    std::cout << "[FAILED!] ISPC is not defined!!!" << std::endl;
    return -1;
    #endif
  }
  else if(a_onGPU)
    save_summary(n_body_gpu(SEED, ITERATIONS), "zout_gpu.json");
  else
    save_summary(n_body_cpu(SEED, ITERATIONS), "zout_cpu.json");

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
    return compute_main(args.hasOption("--gpu"), args.hasOption("--ispc"));
  else
    graphics_main();

  return 0;
}
