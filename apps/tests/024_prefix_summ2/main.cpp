#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <numeric>

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

  std::vector<int> array    (1025*371 + 776); // 1024
  std::vector<int> outArray (array.size());
  for(size_t i=0;i<array.size();i++)
    array[i] = (i % 2 == 0) ? 1 : 0;  

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
  
  pImpl->Resize(array.size());
  pImpl->CommitDeviceData();
  pImpl->PrefixSumm(array.data(), array.size(), outArray.data());

  for(int i=0;i<10;i++)
    std::cout << outArray[i] << " ";
  std::cout << std::endl;

  // check results right now
  //
  std::vector<int> refArray (outArray.size());
  std::exclusive_scan(array.begin(), array.end(), refArray.begin(), 0);
  
  size_t exclusiveDiffId = size_t(-1);
  for(size_t i=0;i<array.size();i++)
  {
    if(refArray[i] != outArray[i])
    {
      exclusiveDiffId = i;
      break;
    }
  }
  
  std::cout << "array size = " << array.size() << std::endl;

  if(exclusiveDiffId == size_t(-1))
  {
    JSONLog::write("exclusive_scan", "PASSED!");
    //std::cout << "exclusive_scan: PASSED!" << std::endl; 
  }
  else
  {
    JSONLog::write("exclusive_scan", "FAILED!");
    //std::cout << "exclusive_scan: FAILED! at " <<  exclusiveDiffId << " " << refArray[exclusiveDiffId] << " != " << outArray[exclusiveDiffId] << std::endl; 
  }

  JSONLog::saveToFile("zout_" + backendName + ".json");

  return 0;
}
