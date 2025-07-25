#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Image2d.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include <JSONLog.hpp>

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<RedPixels> CreateRedPixels_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures RedPixels_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_CUDA
std::shared_ptr<RedPixels> CreateRedPixels_Generated();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::string inputImagePath = "../01_intersectSphere/zout_cpu.bmp";
  int w, h;
  std::vector<uint32_t> inputImageData = LiteImage::LoadBMP(inputImagePath.c_str(), &w, &h);
  if (inputImageData.empty()) {
    throw std::runtime_error("Failed to load image from path: "+inputImagePath);
  }

  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<RedPixels> pImpl = nullptr;
  
  if(onGPU)
  {
    #ifdef USE_VULKAN
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = RedPixels_Generated_ListRequiredDeviceFeatures();
    auto ctx = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl    = CreateRedPixels_Generated(ctx, inputImageData.size());
    #endif
    #ifdef USE_CUDA
    pImpl = CreateRedPixels_Generated();
    #endif
  }
  else
    pImpl = std::make_shared<RedPixels>();
  
  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetMaxDataSize(inputImageData.size());
  pImpl->CommitDeviceData();

  pImpl->ProcessPixels(inputImageData.data(), inputImageData.data(), inputImageData.size());

  JSONLog::write("m_redPixelsNum      : ", pImpl->m_redPixelsNum);
  JSONLog::write("m_otherPixelsNum    : ", pImpl->m_otherPixelsNum);
  JSONLog::write("m_testPixelsAmount  : ", pImpl->m_testPixelsAmount);
  JSONLog::write("m_foundPixels.size(): ", pImpl->m_foundPixels.size());
  JSONLog::write("m_testMin(float)    : ", pImpl->m_testMin);
  JSONLog::write("m_testMax(float)    : ", pImpl->m_testMax);

  JSONLog::saveToFile("zout_"+backendName+".json");
  LiteImage::SaveBMP(("zout_"+backendName+".bmp").c_str(), inputImageData.data(), w, h);
  
  std::cout << std::endl;
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("ProcessPixels", timings);
  std::cout << "ProcessPixels(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "ProcessPixels(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "ProcessPixels(ovrh) = " << timings[3]              << " ms " << std::endl;

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
