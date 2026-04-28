#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstring>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
vk_utils::VulkanDeviceFeatures Derived_Generated_ListRequiredDeviceFeatures();
std::shared_ptr<Base> CreateDerived_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  bool onGPU = args.hasOption("--gpu");

  std::vector<float> array(32);

  std::shared_ptr<Base> pImpl = nullptr;
  
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = Derived_Generated_ListRequiredDeviceFeatures();
    auto ctx = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateDerived_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<Derived>();

  std::string backendName = onGPU ? "gpu" : "cpu";
  
  pImpl->Init(array.size());
  pImpl->CommitDeviceData();

  pImpl->Test(array.data(), unsigned(array.size()));

  JSONLog::write("array", array);

  pImpl->Test_OnlyBase(array.data(), unsigned(array.size()));
  JSONLog::write("array2", array);

  JSONLog::saveToFile("zout_"+backendName+".json");
  
  pImpl = nullptr;
  vk_utils::globalContextDestroy();  

  return 0;
}
