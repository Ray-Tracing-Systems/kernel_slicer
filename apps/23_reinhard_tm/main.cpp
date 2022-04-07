#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "reinhard.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp
void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);

#include "vk_context.h"
std::shared_ptr<ReinhardTM> CreateReinhardTM_Generated(int a_w, int a_h, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated) ;

int main(int argc, const char** argv)
{

  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> hdrData;
  int w,h;
  
  if(!LoadHDRImageFromFile("../images/kitchen.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'kitchen.hdr' " << std::endl;
    return 0;
  }
  
  bool onGPU = true;

  std::shared_ptr<ReinhardTM> pImpl = nullptr;
  
  if(onGPU)
  {
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, 0);
    pImpl    = CreateReinhardTM_Generated(w,h, ctx,w*h);
  }
  else
    pImpl = std::make_shared<ReinhardTM>(w,h);

  std::vector<uint> ldrData(w*h);

  // Reinhard tm algorithm
  //
  pImpl->CommitDeviceData();
  pImpl->Run(w,h, hdrData.data(), ldrData.data());
  
  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);


  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Run", timings);
  std::cout << "Run(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "Run(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "Run(ovrh) = " << timings[3]              << " ms " << std::endl; 
  
  return 0;
}
