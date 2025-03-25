#include <vector>
#include <memory>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>


#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<SimpleTest> CreateSimpleTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif
#ifdef USE_CUDA
std::shared_ptr<SimpleTest> CreateSimpleTest_Generated()
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> outputArray(8);
  for(size_t i=0;i<outputArray.size();i++)
    outputArray[i] = 0;

  std::shared_ptr<SimpleTest> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    #ifdef USE_CUDA
    pImpl = CreateSimpleTest_Generated();
    #endif
    #ifdef USE_VULKAN
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    std::vector<const char*> requiredExtensions;
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateSimpleTest_Generated(ctx, outputArray.size());
    #endif
  }
  else
    pImpl = std::make_shared<SimpleTest>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->CalcAndAccum(outputArray.size(), outputArray.data());
  
  float outputArray2[8] = {};
  for(int i=0;i<8;i++) {
    std::cout << i << "\t" << outputArray[i] << std::endl;
    outputArray2[i] = outputArray[i];
  }

  JSONLog::write("array", outputArray2);
  JSONLog::saveToFile("zout_"+backendName+".json");
  
  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif

  return 0;
}
