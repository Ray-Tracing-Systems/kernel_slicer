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

#define LAYOUT_STD140
#include "LiteMath.h"

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

  std::vector<TestData> array    (1024*256); // 1024
  std::vector<TestData> outArray (array.size());
  for(size_t i=0;i<array.size();i++)
  {
    TestData test;
    test.key = rand() % 2000;
    test.val = float(i);
    test.dummy1 = 2*i;
    test.dummy2 = 3*i;
    array[i] = test;  
  }
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
  
  pImpl->Reserve(array.size());
  pImpl->CommitDeviceData();
  pImpl->Sort(array.data(), array.size(), outArray.data());

  for(int i=0;i<10;i++)
    std::cout << "(" << outArray[i].key << ";" << outArray[i].val << "), ";
  std::cout << std::endl;

  // check results right now
  //
  std::vector<TestData> refArray = array;
  std::sort(refArray.begin(), refArray.end(), [](TestData a, TestData b) { return a.key < b.key; });
  
  size_t diffId = size_t(-1);
  for(size_t i=0;i<array.size();i++)
  {
    if(refArray[i].key != outArray[i].key) 
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
