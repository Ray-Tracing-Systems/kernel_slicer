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
std::shared_ptr<PtrCast> CreatePtrCast_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures PtrCast_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000;
  std::vector<float> in(SIZE), out(SIZE*8, 0.0f);
  for(uint32_t i = 0; i < SIZE; i++)
    in[i] = float(i % 23) - 11.0f;

  std::shared_ptr<PtrCast> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = PtrCast_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreatePtrCast_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<PtrCast>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->m_scratch.resize(SIZE*4);
  pImpl->CommitDeviceData();
  pImpl->Run(in.data(), out.data(), SIZE);
  std::cout << "out[0..3] = " << out[0] << " " << out[1] << " " << out[2] << " " << out[3] << std::endl; // -22*0.03125, 1, 0.03125, -1

  JSONLog::write("out", out);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
