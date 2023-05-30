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

  std::vector<int> array    (66450); // 1024
  std::vector<int> outArray (array.size());
  std::vector<int> outArray2(array.size());
  //for(size_t i=0;i<array.size();i++)
  //  array[i] = i + 1;
  for(size_t i=0;i<array.size();i++)
    array[i] = 1;    

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
  pImpl->PrefixSumm(array.data(), array.size(), outArray.data(), outArray2.data());

  for(int i=0;i<10;i++)
    std::cout << outArray[i] << " ";
  std::cout << std::endl;

  for(int i=0;i<10;i++)
    std::cout << outArray2[i] << " ";
  std::cout << std::endl;
  
  // check results right now
  //
  std::vector<int> refArray (outArray.size());
  std::vector<int> refArray2(outArray.size());
 
  std::exclusive_scan(array.begin(), array.end(), refArray.begin(), 0);
  std::inclusive_scan(array.begin(), array.end(), refArray2.begin(), std::plus<int>(), 0);
  
  size_t exclusiveDiffId = size_t(-1);
  size_t inclusiveDiffId = size_t(-1);

  for(size_t i=0;i<array.size();i++)
  {
    if(refArray[i] != outArray[i])
    {
      exclusiveDiffId = i;
      break;
    }
  }

  for(size_t i=0;i<array.size();i++)
  {
    if(refArray2[i] != outArray2[i])
    {
      inclusiveDiffId = i;
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
    std::ofstream fout("z_ex_scan.txt");
    for(size_t i=0;i<outArray.size();i++)
      fout << i << "\t" << outArray[i] << std::endl;
  }

  if(inclusiveDiffId == size_t(-1))
  {
    JSONLog::write("inclusive_scan", "PASSED!");
    //std::cout << "inclusive_scan: PASSED!" << std::endl; 
  }
  else
  {
    JSONLog::write("inclusive_scan", "FAILED!");
    //std::cout << "inclusive_scan: FAILED! at " <<  inclusiveDiffId << " " << refArray[inclusiveDiffId] << " != " << outArray[inclusiveDiffId] << std::endl; 
  }

  JSONLog::saveToFile("zout_" + backendName + ".json");

  return 0;
}
