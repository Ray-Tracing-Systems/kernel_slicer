#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<RedPixels> CreateRedPixels_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP("../01_intersectSphere/zout_cpu.bmp", &w, &h);
  
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<RedPixels> pImpl = nullptr;
  
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateRedPixels_Generated(ctx, inputImageData.size());
  }
  else
    pImpl = std::make_shared<RedPixels>();

  pImpl->SetMaxDataSize(inputImageData.size());
  pImpl->CommitDeviceData();

  pImpl->ProcessPixels(inputImageData.data(), inputImageData.data(), inputImageData.size());

  std::cout << "m_redPixelsNum     = " << pImpl->m_redPixelsNum << std::endl;
  std::cout << "m_otherPixelsNum   = " << pImpl->m_otherPixelsNum << std::endl;
  std::cout << "m_testPixelsAmount = " << pImpl->m_testPixelsAmount << std::endl;
  std::cout << "m_foundPixels_size = " << pImpl->m_foundPixels.size() << std::endl;
  std::cout << "m_testMin(float)   = " << pImpl->m_testMin << std::endl;
  std::cout << "m_testMax(float)   = " << pImpl->m_testMax << std::endl;
  std::cout << "found "                << pImpl->m_foundPixels.size() << " red pixels" << std::endl;

  if(onGPU)
    SaveBMP("z_out_gpu.bmp", inputImageData.data(), w, h);
  else
    SaveBMP("z_out_cpu.bmp", inputImageData.data(), w, h);
  
  std::cout << std::endl;
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("ProcessPixels", timings);
  std::cout << "ProcessPixels(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "ProcessPixels(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "ProcessPixels(ovrh) = " << timings[3]              << " ms " << std::endl;
  return 0;
}
