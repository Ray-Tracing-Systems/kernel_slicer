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
std::shared_ptr<ReinhardTM> CreateReinhardTM_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
std::shared_ptr<ReinhardTM> CreateReinhardTM_ISPC();

int main(int argc, const char** argv)
{
  std::vector<float> hdrData;
  int w,h;
  
  if(!LoadHDRImageFromFile("kitchen.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'kitchen.hdr' " << std::endl;
    return 0;
  }

  std::vector<uint> ldrData(w*h);

  bool onGPU  = true;
  bool isISPC = false;

  std::shared_ptr<ReinhardTM> pImpl = nullptr;

  //auto pImpl = std::make_shared<ReinhardTM>();
  if(onGPU)
  {
    auto ctx   = vk_utils::globalContextGet(false, 0);
    pImpl = CreateReinhardTM_Generated(ctx, w*h);
  }
  else if(isISPC)
  {
    pImpl = CreateReinhardTM_ISPC();
  }
  else
    pImpl = std::make_shared<ReinhardTM>();

  pImpl->CommitDeviceData();
  pImpl->Run(w, h, (const float4*)hdrData.data(), ldrData.data());

  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else if(isISPC)
    SaveBMP("zout_ispc.bmp", ldrData.data(), w, h);
  else
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);

  std::cout << "whitePoint = " << pImpl->getWhitePoint() << std::endl;

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Run", timings);
  std::cout << "Run(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "Run(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "Run(ovrh) = " << timings[3]              << " ms " << std::endl;

  return 0;
}
