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
std::shared_ptr<SimpleTest> CreateSimpleTest_Generated();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> outputArray(4);
  std::vector<float> inputArray(256*256);
  for(size_t i=0;i<outputArray.size();i++)
    outputArray[i] = 0;
  for(size_t i=0;i<inputArray.size();i++)
    inputArray[i] = 0.01f*float(i);

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
  pImpl->CalcAndAccum(inputArray.data(), inputArray.size(), outputArray.data());
  
  for(int i=0;i<outputArray.size();i++) 
    std::cout << i << "\t" << outputArray[i] << std::endl;

  JSONLog::write("array", outputArray);
  JSONLog::saveToFile("zout_"+backendName+".json");
  
  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif

  return 0;
}
