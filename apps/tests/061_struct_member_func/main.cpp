#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<StructMembers> CreateStructMembers_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures StructMembers_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 32*32;
  std::vector<SphereLight> lights(4);
  for(uint32_t i=0;i<lights.size();i++)
  {
    lights[i].posAndRadius = float4(float(i*8), float(i*5), 4.0f + float(i), 0.5f + 0.25f*float(i));
    lights[i].color        = float4(1.0f, 1.0f, 1.0f, 10.0f*float(i+1));
  }
  std::vector<float> out(SIZE);

  std::shared_ptr<StructMembers> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = StructMembers_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateStructMembers_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<StructMembers>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetLights(lights);
  pImpl->CommitDeviceData();
  pImpl->Run(out.data(), SIZE);

  std::cout << "out[0] = " << out[0] << ", out[100] = " << out[100] << std::endl;

  JSONLog::write("out", out);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
