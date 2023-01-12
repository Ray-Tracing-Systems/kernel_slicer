#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "reinhard.h"
#include "ArgParser.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp
void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);

#include "vk_context.h"
std::shared_ptr<ReinhardTM> CreateReinhardTM_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
#ifdef USE_ISPC
std::shared_ptr<ReinhardTM> CreateReinhardTM_ISPC();
#endif

int main(int argc, const char** argv)
{
  std::vector<float> hdrData;
  int w,h;
  
  if(!LoadHDRImageFromFile("../images/nancy_church_2.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file '../images/nancy_church_2.hdr' " << std::endl;
    return 0;
  }

  std::vector<uint> ldrData(w*h);
  
  ArgParser args(argc, argv);
  bool onGPU  = args.hasOption("--gpu");
  bool isISPC = args.hasOption("--ispc");

  std::shared_ptr<ReinhardTM> pImpl = nullptr;
  if(onGPU)
  {
    auto ctx   = vk_utils::globalContextGet(false, 0);
    pImpl = CreateReinhardTM_Generated(ctx, w*h);
  }
  #ifdef USE_ISPC
  else if(isISPC)
    pImpl = CreateReinhardTM_ISPC();
  #endif
  else
    pImpl = std::make_shared<ReinhardTM>();

  pImpl->CommitDeviceData();
  pImpl->Run(w, h, hdrData.data(), ldrData.data()); // (const float4*)

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
