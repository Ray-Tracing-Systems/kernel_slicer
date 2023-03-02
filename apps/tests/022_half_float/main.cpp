#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
std::shared_ptr<HalfTest> CreateHalfTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> array(4*14);

  std::shared_ptr<HalfTest> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateHalfTest_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<HalfTest>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->Test(array.data(), 4);

  for(int i=0;i<4*14;i++)
    std::cout << i << "\t" << array[i] << std::endl;
  
  float outArray[4*14] = {};
  memcpy(outArray, array.data(), sizeof(outArray));

  JSONLog::write("array", outArray);
  JSONLog::saveToFile("zout_"+backendName+".json");

  return 0;
}
