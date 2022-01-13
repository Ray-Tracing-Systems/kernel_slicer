#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

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
  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<TestCombinedImage> pImpl = nullptr;
  if(onGPU)
  {
    //unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    //auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    //pImpl = CreateDenoise_Generated(ctx, w*h);
  }
  else
    pImpl = std::make_shared<TestCombinedImage>();

  pImpl->CommitDeviceData();

  pImpl->Run(width, height, color.data());

  if(onGPU)
    SaveBMP("zout_gpu.bmp", color.data(), width, height);  
  else
    SaveBMP("zout_cpu.bmp", color.data(), width, height);  

  return 0;
}
