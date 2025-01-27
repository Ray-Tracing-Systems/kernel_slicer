#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <algorithm>
#include <numeric>
#include <cstdlib>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "LiteMath.h"
using LiteMath::uint2;

#include "vk_context.h"
std::shared_ptr<Sorter> CreateSorter_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  srand(777);

  std::vector<uint2> array    (1024*256); // 1024
  std::vector<uint2> outArray (array.size());
  for(size_t i=0;i<array.size();i++)
    array[i] = uint2(rand() % 2000, i);  

  std::shared_ptr<Sorter> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateSorter_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<Sorter>();
  std::string backendName = onGPU ? "gpu" : "cpu";
  
  pImpl->CommitDeviceData();
  pImpl->Sort(array.data(), array.size(), outArray.data());

  for(int i=0;i<10;i++)
    std::cout << "(" << outArray[i].x << ";" << outArray[i].y << "), ";
  std::cout << std::endl;

  // check results right now
  //
  std::vector<uint2> refArray = array;
  std::sort(refArray.begin(), refArray.end(), [](uint2 a, uint2 b) { return a.x < b.x; });
  
  size_t diffId = size_t(-1);
  for(size_t i=0;i<array.size();i++)
  {
    if(refArray[i].x != outArray[i].x) // || refArray[i].y != outArray[i].y
    {
      diffId = i;
      break;
    }
  }
  
  std::cout << "array size = " << array.size() << std::endl;

  if(diffId == size_t(-1))
  {
    JSONLog::write("sort", "PASSED!");
    //std::cout << "sort: PASSED!" << std::endl; 
  }
  else
  {
    JSONLog::write("sort", "FAILED!");
    //std::cout << "sort: FAILED! at " <<  diffId << " " << std::endl; // << refArray[diffId] << " != " << outArray[diffId] << std::endl; 
  }

  JSONLog::saveToFile("zout_" + backendName + ".json");

  return 0;
}
