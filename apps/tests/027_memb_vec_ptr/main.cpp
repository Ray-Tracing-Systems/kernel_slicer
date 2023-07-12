#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <memory>

#include "test_class.h"

#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
std::shared_ptr<TestVecDataAccessFromMember> CreateTestVecDataAccessFromMember_Generated(size_t a_size, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  
  const size_t size = 256;
  std::vector<int> color(size);
 
  ArgParser args(argc, argv);
  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<TestVecDataAccessFromMember> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestVecDataAccessFromMember_Generated(size, ctx, size);
  }
  else
    pImpl = std::make_shared<TestVecDataAccessFromMember>(size);

  pImpl->CommitDeviceData();

  pImpl->Run(size, color.data());
  
  std::string backendName = onGPU ? "gpu" : "cpu";
  JSONLog::write("array", color);
  JSONLog::saveToFile("zout_"+backendName+".json");

  return 0;
}
