#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<TestCombinedImage> CreateTestCombinedImage_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures TestCombinedImage_Generated_ListRequiredDeviceFeatures();
#ifdef USE_ISPC
std::shared_ptr<TestCombinedImage> CreateTestCombinedImage_ISPC();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  
  uint32_t width  = 512;
  uint32_t height = 512;
  std::vector<uint32_t> color(width*height);
 
  ArgParser args(argc, argv);
  bool onGPU  = args.hasOption("--gpu");
  bool onISPC = args.hasOption("--ispc");

  std::shared_ptr<TestCombinedImage> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0); //..
    auto deviceFeatures = TestCombinedImage_Generated_ListRequiredDeviceFeatures();
    auto ctx            = vk_utils::globalContextInit(deviceFeatures, enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestCombinedImage_Generated(ctx, width*height);
  }
  #ifdef USE_ISPC
  else if(onISPC)
    pImpl = CreateTestCombinedImage_ISPC();
  #endif
  else
    pImpl = std::make_shared<TestCombinedImage>();

  pImpl->CommitDeviceData();

  pImpl->Run(width, height, color.data());

  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", color.data(), width, height);  
  else if(onISPC)
    LiteImage::SaveBMP("zout_ispc.bmp", color.data(), width, height);  
  else
    LiteImage::SaveBMP("zout_cpu.bmp", color.data(), width, height);  
  
  pImpl = nullptr;
  vk_utils::globalContextDestroy();  

  return 0;
}
