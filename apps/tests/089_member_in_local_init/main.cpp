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
std::shared_ptr<LocalInit> CreateLocalInit_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures LocalInit_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<LocalInit> CreateLocalInit_ISPC();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t PITCH = 64, ROWS = 48, SIZE = PITCH*ROWS;
  std::vector<float> in(SIZE), out(SIZE, 0.0f);
  for(uint32_t i = 0; i < SIZE; i++)
    in[i] = float(i % 29) - 14.0f;

  std::shared_ptr<LocalInit> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  bool onISPC = args.hasOption("--ispc");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = LocalInit_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateLocalInit_Generated(ctx, SIZE);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateLocalInit_ISPC();
  else
  #endif
    pImpl = std::make_shared<LocalInit>();

  std::string backendName = onGPU ? "gpu" : (onISPC ? "ispc" : "cpu");

  pImpl->SetWindow(5, 3, PITCH); // window 40x30 at (5, 3)
  pImpl->CommitDeviceData();
  pImpl->Run(40, 30, in.data(), out.data(), SIZE);
  std::cout << "out[3*64+5] = " << out[3*PITCH+5] << ", out[0] = " << out[0] << std::endl;

  JSONLog::write("out", out);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
