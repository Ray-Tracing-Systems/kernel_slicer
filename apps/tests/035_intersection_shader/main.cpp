#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <string>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(int w, int h, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

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
    pImpl = CreateTestClass_Generated(TestClass::WIN_WIDTH, TestClass::WIN_HEIGHT, ctx, TestClass::WIN_WIDTH*TestClass::WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<TestClass>(TestClass::WIN_WIDTH, TestClass::WIN_HEIGHT);
  
  std::vector<uint> pixelData(TestClass::WIN_WIDTH*TestClass::WIN_HEIGHT);  
  
  pImpl->InitScene(60,25);
  pImpl->CommitDeviceData();  
  pImpl->BFRT_ReadAndComputeBlock(TestClass::WIN_WIDTH, TestClass::WIN_HEIGHT, pixelData.data(), 1);
    
  if(onGPU)
    SaveBMP("zout_gpu.bmp", pixelData.data(), TestClass::WIN_WIDTH, TestClass::WIN_HEIGHT);
  else
    SaveBMP("zout_cpu.bmp", pixelData.data(), TestClass::WIN_WIDTH, TestClass::WIN_HEIGHT);
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("BFRT_ReadAndCompute", timings);
  std::cout << "BFRT_ReadAndCompute(exec) = " << timings[0] << " ms " << std::endl;
  

  pImpl = nullptr;
  return 0;
}