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
std::shared_ptr<TestClass> CreateTestClass_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> array(16);

  std::shared_ptr<TestClass> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestClass_Generated(ctx, array.size());
  }
  else
    pImpl = std::make_shared<TestClass>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->Test(array.data(), unsigned(array.size()));

  for(int i=0;i<16;i++)
    std::cout << i << ":\t" << array[i] << std::endl;
  
  //float  outArray2[16] = {};
  //
  //for(int i=0;i<4;i++) 
  //{
  //  outArray2[i*4+0]  = array[i][0];
  //  outArray2[i*4+1]  = array[i][1];
  //  outArray2[i*4+2]  = array[i][2];
  //  outArray2[i*4+3]  = array[i][3];
  //}

  JSONLog::write("array", array);
  JSONLog::saveToFile("zout_"+backendName+".json");

  return 0;
}
