#include <iostream>
#include <fstream>

#include "test_class.h"
#include "Image2d.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(int a_maxThreads, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#ifdef USE_ISPC
std::shared_ptr<TestClass> CreateTestClass_ISPC(int a_maxThreads);
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
  std::vector<uint32_t> packedXY(WIN_WIDTH*WIN_HEIGHT);
  std::vector<float4>   realColor(WIN_WIDTH*WIN_HEIGHT);
  
  std::shared_ptr<TestClass> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU  = args.hasOption("--gpu");
  bool isISPC = args.hasOption("--ispc");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestClass_Generated( WIN_WIDTH*WIN_HEIGHT, ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  #ifdef USE_ISPC
  else if(isISPC)
    pImpl = CreateTestClass_ISPC(WIN_WIDTH*WIN_HEIGHT);
  #endif
  else
    pImpl = std::make_shared<TestClass>(WIN_WIDTH*WIN_HEIGHT);
  
  pImpl->LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.vsgf");
  pImpl->CommitDeviceData();

  // remember pitch-linear (x,y) for each thread to make our threading 1D
  //
  pImpl->PackXY(WIN_WIDTH, WIN_HEIGHT, packedXY.data());
  pImpl->CastSingleRay(WIN_WIDTH*WIN_HEIGHT, packedXY.data(), pixelData.data());
  
  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else if(isISPC)
    LiteImage::SaveBMP("zout_ispc.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    LiteImage::SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("CastSingleRay", timings);
  std::cout << "CastSingleRay(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "CastSingleRay(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "CastSingleRay(ovrh) = " << timings[3]              << " ms " << std::endl;

  pImpl = nullptr;
  vk_utils::globalContextDestroy();
  return 0;
}

