#include <iostream>
#include <fstream>
#include <vector>
#include <memory>

#include "test_class.h"
#include "Bitmap.h"

#include "vk_context.h"
std::shared_ptr<TestClass> CreateTestClass_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

#define MEASURE_TIME
#ifdef MEASURE_TIME
#include "test_class_generated.h"
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  
  bool onGPU = true;
  std::shared_ptr<TestClass> pImpl = nullptr;
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers);
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
  
  #ifdef MEASURE_TIME
  auto pGPUImpl = dynamic_cast<TestClass_Generated*>(pImpl.get());
  if(pGPUImpl != nullptr)
  {
    auto timings = pGPUImpl->GetMainFuncExecutionTime();
    std::cout << "MainFunc(exec) = " << timings.msExecuteOnGPU                      << " ms " << std::endl;
    std::cout << "MainFunc(copy) = " << timings.msCopyToGPU + timings.msCopyFromGPU << " ms " << std::endl;
    std::cout << "MainFunc(ovrh) = " << timings.msAPIOverhead << " ms " << std::endl;
  }
  #endif
  pImpl = nullptr;
  return 0;
}
