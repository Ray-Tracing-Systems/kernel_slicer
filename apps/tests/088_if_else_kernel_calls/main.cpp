#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<IfElseCalls> CreateIfElseCalls_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures IfElseCalls_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<IfElseCalls> CreateIfElseCalls_ISPC();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000;
  std::vector<float> in(SIZE);
  for(uint32_t i = 0; i < SIZE; i++)
    in[i] = float(i % 23) - 11.0f;

  std::shared_ptr<IfElseCalls> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  bool onISPC = args.hasOption("--ispc");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = IfElseCalls_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateIfElseCalls_Generated(ctx, SIZE);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateIfElseCalls_ISPC();
  else
  #endif
    pImpl = std::make_shared<IfElseCalls>();

  std::string backendName = onGPU ? "gpu" : (onISPC ? "ispc" : "cpu");

  pImpl->CommitDeviceData();
  for(uint32_t mode = 0; mode < 3; mode++)
  {
    std::vector<float> out(SIZE, 0.0f);
    pImpl->SetMode(mode);
    pImpl->UpdateMembersPlainData();
    pImpl->Run(in.data(), out.data(), SIZE);
    std::cout << "mode " << mode << ": out[0] = " << out[0] << ", out[1] = " << out[1] << std::endl; // expected: -10 -9, -33 -30, 10 9
    JSONLog::write("out" + std::to_string(mode), out);
  }
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
