#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>
#include <cstdlib>

#include "test_class.h"

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include <JSONLog.hpp>

#include "vk_context.h"
std::shared_ptr<BoxMinMax> CreateBoxMinMax_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const size_t N = 2399;
  //const size_t N = 257;

  std::vector<float4> points(N);
  //for(auto& p : points)
  //  p = float4(rand() % 2000 - 1000, rand() % 200 - 100, rand() % 20 - 10, (rand() % 2000 - 1000)*0.00001f);
  for(int i=0;i<int(points.size());i++)
    points[i] = float4(float(i) - 100, float(-i), rand() % 200 - 100 - 5, float(i/2) + 10);

  float4 minMax[2] = {};

  ArgParser args(argc, argv);
  bool onGPU = args.hasOption("--gpu");
  std::shared_ptr<BoxMinMax> pImpl = nullptr;
  
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateBoxMinMax_Generated(ctx, points.size());
  }
  else
    pImpl = std::make_shared<BoxMinMax>();
  
  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->ProcessPoints(points.data(), points.size(), minMax);

  JSONLog::write("min.x", minMax[0].x);
  JSONLog::write("min.y", minMax[0].y);
  JSONLog::write("min.z", minMax[0].z);
  JSONLog::write("min.w", minMax[0].w);
 
  JSONLog::write("max.x", minMax[1].x);
  JSONLog::write("max.y", minMax[1].y);
  JSONLog::write("max.z", minMax[1].z);
  JSONLog::write("max.w", minMax[1].w);

  JSONLog::saveToFile("zout_"+backendName+".json");
  std::cout << std::endl;
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("ProcessPoints", timings);
  std::cout << "ProcessPoints(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "ProcessPoints(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "ProcessPoints(ovrh) = " << timings[3]              << " ms " << std::endl;
  return 0;
}
