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
std::shared_ptr<PrefSummTest> CreatePrefSummTest_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<int> array    (1024);
  std::vector<int> outArray (1024);
  std::vector<int> outArray2(1024);
  for(size_t i=0;i<array.size();i++)
    array[i] = i + 1;

  std::shared_ptr<PrefSummTest> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreatePrefSummTest_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<PrefSummTest>();
  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->PrefixSumm(array.data(), array.size(), 
                    outArray.data(), outArray2.data());

  for(int i=0;i<10;i++)
    std::cout << outArray[i] << " ";
  std::cout << std::endl;

  for(int i=0;i<10;i++)
    std::cout << outArray2[i] << " ";
  std::cout << std::endl;

  //JSONLog::write("array", outArray);
  //JSONLog::write("array2", outArray2);
  //JSONLog::saveToFile("zout_" + backendName + ".json");

  return 0;
}
