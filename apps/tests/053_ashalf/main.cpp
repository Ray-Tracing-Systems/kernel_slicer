#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <memory>

#include "test_class.h"

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<Test2D> CreateTest2D_Generated(size_t a_size, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures Test2D_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  
  const size_t size = 8;
  std::vector<float4> color(size);
 
  ArgParser args(argc, argv);
  bool onGPU = args.hasOption("--gpu"); 

  std::shared_ptr<Test2D> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = Test2D_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTest2D_Generated(size, ctx, size);
  }
  else
  #endif
    pImpl = std::make_shared<Test2D>(size);

  pImpl->CommitDeviceData();

  pImpl->Run(size, color.data());
  

  std::vector<float> outData(color.size()*4);
  memcpy(outData.data(), color.data(), outData.size()*sizeof(float));

  std::string backendName = onGPU ? "gpu" : "cpu";
  JSONLog::write("array", outData);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
