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
std::shared_ptr<InlineFuncs> CreateInlineFuncs_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures InlineFuncs_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t WIDTH = 37, HEIGHT = 21;
  std::vector<uint32_t> out(WIDTH*HEIGHT*2);

  std::shared_ptr<InlineFuncs> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = InlineFuncs_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateInlineFuncs_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<InlineFuncs>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->Run(out.data(), WIDTH, HEIGHT);

  int failed = 0;
  for(size_t i=0;i<out.size()/2;i++)
    if(out[i*2+1] == 0xFFFFFFFFu)
      failed++;
  std::cout << "out[2] = " << out[2] << ", out[3] = " << out[3] << ", failed = " << failed << std::endl;

  JSONLog::write("out", out);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return (failed == 0) ? 0 : 1;
}
