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
std::shared_ptr<LoopBreak> CreateLoopBreak_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures LoopBreak_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000, MAX_ITERS = 100; // some sequences are longer than MAX_ITERS and leave the loop by its condition
  std::vector<uint32_t> start(SIZE), steps(SIZE, 0), peak(SIZE, 0);
  for(uint32_t i = 0; i < SIZE; i++)
    start[i] = i + 1;

  std::shared_ptr<LoopBreak> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = LoopBreak_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateLoopBreak_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<LoopBreak>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->CollatzBlock(SIZE, MAX_ITERS, start.data(), steps.data(), peak.data(), 1);

  std::cout << "steps[26] = " << steps[26] << ", peak[26] = " << peak[26] << ", steps[999] = " << steps[999] << std::endl;

  JSONLog::write("steps", steps);
  JSONLog::write("peak", peak);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
