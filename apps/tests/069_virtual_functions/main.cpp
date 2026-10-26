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
std::shared_ptr<VirtualShapes> CreateVirtualShapes_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures VirtualShapes_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t WIDTH = 64, HEIGHT = 40;
  std::vector<float> image(WIDTH*HEIGHT, -1.0f);

  std::shared_ptr<VirtualShapes> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = VirtualShapes_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateVirtualShapes_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<VirtualShapes>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->InitShapes();
  pImpl->CommitDeviceData();
  pImpl->Render(WIDTH, HEIGHT, image.data());

  std::cout << "image[8*64+12] = " << image[8*64+12] << ", image[3*64+3] = " << image[3*64+3] << std::endl;

  JSONLog::write("image", image);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
