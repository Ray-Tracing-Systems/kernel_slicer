#include <iostream>
#include <fstream>
#include <memory>

#include "test_class.h"
#include "Image2d.h"
#include "ArgParser.h"

#include "vk_context.h"
#include "test_class_generated.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(int a_maxThreads, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);


int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::cout << "sizeof(IMaterial) = " << sizeof(IMaterial) << std::endl; 

  std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
  std::vector<uint32_t> packedXY(WIN_WIDTH*WIN_HEIGHT);
  std::vector<float4>   realColor(WIN_WIDTH*WIN_HEIGHT);
  
  std::shared_ptr<TestClass> pImpl = nullptr;
  ArgParser args(argc, argv);
  
  //bool onGPU = false;
  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    std::vector<const char*> requiredExtensions;
    auto deviceFeatures = TestClass_Generated::ListRequiredDeviceFeatures(requiredExtensions);
    auto ctx            = vk_utils::globalContextInit(requiredExtensions, enableValidationLayers, a_preferredDeviceId, &deviceFeatures);
    //unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    //auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateTestClass_Generated( WIN_WIDTH*WIN_HEIGHT, ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<TestClass>(WIN_WIDTH*WIN_HEIGHT);
  
  pImpl->LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.bvh", "../10_virtual_func_rt_test1/cornell_collapsed.vsgf");
  pImpl->CommitDeviceData();

  // remember pitch-linear (x,y) for each thread to make our threading 1D
  //
  pImpl->PackXYBlock(WIN_WIDTH, WIN_HEIGHT, packedXY.data(), 1);
  pImpl->CastSingleRayBlock(WIN_HEIGHT*WIN_HEIGHT, packedXY.data(), pixelData.data(), 1);
  
  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    LiteImage::SaveBMP("zout_cpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  std::cout << "start NaivePathTraceBlock ... " << std::endl;
  // now test path tracing
  //
  const int PASS_NUMBER = 256;
  pImpl->NaivePathTraceBlock(WIN_HEIGHT*WIN_HEIGHT, 6, packedXY.data(), realColor.data(), PASS_NUMBER);
  
  const float normConst = 1.0f/float(PASS_NUMBER);
  const float invGamma  = 1.0f / 2.2f;

  for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
  {
    float4 color = realColor[i]*normConst;
    color.x      = std::pow(color.x, invGamma);
    color.y      = std::pow(color.y, invGamma);
    color.z      = std::pow(color.z, invGamma);
    color.w      = 1.0f;
    pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
  }

  if(onGPU)
    LiteImage::SaveBMP("zout_gpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  else
    LiteImage::SaveBMP("zout_cpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("NaivePathTraceBlock", timings);
  std::cout << "NaivePathTraceBlock(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "NaivePathTraceBlock(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "NaivePathTraceBlock(ovrh) = " << timings[3]              << " ms " << std::endl;

  pImpl->GetExecutionTime("NaivePathTraceMega", timings);
  std::cout << "NaivePathTraceMega(avg) = " << timings[0]*PASS_NUMBER << " ms " << std::endl;
  std::cout << "NaivePathTraceMega(min) = " << timings[1]*PASS_NUMBER << " ms " << std::endl;
  std::cout << "NaivePathTraceMega(max) = " << timings[2]*PASS_NUMBER << " ms " << std::endl;

  return 0;
}
