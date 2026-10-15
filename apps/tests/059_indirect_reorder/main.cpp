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
std::shared_ptr<IndirectReorder> CreateIndirectReorder_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures IndirectReorder_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t WIDTH = 300, HEIGHT = 20; // width >> height to catch swapped dispatch sizes
  std::vector<uint32_t> out(WIDTH*HEIGHT, 0);

  std::shared_ptr<IndirectReorder> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = IndirectReorder_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateIndirectReorder_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<IndirectReorder>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetSize(WIDTH, HEIGHT);
  pImpl->CommitDeviceData();
  pImpl->Run(out.data(), uint32_t(out.size()));

  int failed = 0;
  for(uint32_t y=0;y<HEIGHT;y++)
  {
    for(uint32_t x=0;x<WIDTH;x++)
    {
      const uint32_t expected = (y < HEIGHT-3) ? (x + y*1000 + 1)*2 : 0;
      if(out[y*WIDTH+x] != expected)
        failed++;
    }
  }
  std::cout << "m_height = " << pImpl->m_height << ", failed = " << failed << std::endl;

  JSONLog::write("out", out);
  JSONLog::write("height", pImpl->m_height);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return (failed == 0) ? 0 : 1;
}
