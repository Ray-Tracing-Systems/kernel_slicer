#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

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
    pImpl = CreateTestClass_Generated(1024, 1024, ctx, 1024*1024);
  }
  else
    pImpl = std::make_shared<TestClass>(1024,1024);

  pImpl->InitBoxesAndTris(60,25); // init with some max number
  
  // evaluate timings

  pImpl->SetBoxTrisNum(50,25);
  pImpl->CommitDeviceData();

  std::vector<uint> pixelData(WIN_WIDTH*WIN_HEIGHT);  
  pImpl->BFRT_ReadAndComputeBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);
  
  if(onGPU)
    SaveBMP("zout_gpu_v1.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu_v1.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("BFRT_ReadAndCompute", timings);
  std::cout << "BFRT_ReadAndCompute(exec) = " << timings[0] << " ms " << std::endl;
  
  pImpl->BFRT_ComputeBlock(WIN_WIDTH, WIN_HEIGHT, pixelData.data(), 1);

  if(onGPU)
    SaveBMP("zout_gpu_v2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    SaveBMP("zout_cpu_v2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  pImpl->GetExecutionTime("BFRT_Compute", timings);
  std::cout << "BFRT_Compute(exec) = " << timings[0] << " ms " << std::endl;

  // print timings and max M(Rays/sec)

  pImpl = nullptr;
  return 0;
}