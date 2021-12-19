#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<TestClass> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestClass_Generated(ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<TestClass>();

  pImpl->CommitDeviceData();

  std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);  
  pImpl->MainFuncBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
  
  if(onGPU)
    SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("MainFunc", timings);
  std::cout << "MainFunc(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "MainFunc(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "MainFunc(ovrh) = " << timings[3]              << " ms " << std::endl;
  
  pImpl = nullptr;
  return 0;
}