#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>
#include <cmath>

#include "test_class.h"
#include "Bitmap.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data);   // defined in imageutils.cpp
bool LoadLDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<int32_t>& a_data); // defined in imageutils.cpp

std::shared_ptr<Denoise> CreateDenoise_Generated();

#define MEASURE_TIME
#ifdef MEASURE_TIME
#include "test_class_generated.h"
#endif

int main(int argc, const char** argv)
{
  std::vector<float>   hdrData;
  std::vector<int32_t> texColor;
  std::vector<int32_t> normal;
  std::vector<float>   depth;

  int w, h, w2, h2, w3, h3, w4, h4;
  
  bool hasError = false;

  if(!LoadHDRImageFromFile("../images/WasteWhite_1024sample.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'WasteWhite_1024sample.hdr' " << std::endl;
    hasError = true;
  }

  if(!LoadHDRImageFromFile("../images/WasteWhite_depth.hdr", &w2, &h2, depth))
  {
    std::cout << "can't open input file 'WasteWhite_depth.hdr' " << std::endl;
    hasError = true;
  }

  if(!LoadLDRImageFromFile("../images/WasteWhite_diffcolor.png", &w3, &h3, texColor))
  {
    std::cout << "can't open input file 'WasteWhite_diffcolor.png' " << std::endl;
    hasError = true;
  }

  if(!LoadLDRImageFromFile("../images/WasteWhite_normals.png", &w4, &h4, normal))
  {
    std::cout << "can't open input file 'WasteWhite_normals.png' " << std::endl;
    hasError = true;
  }

  if(w != w2 || h != h2)
  {
    std::cout << "size source image and depth pass not equal.' " << std::endl;
    hasError = true;
  }
  
  if(w != w3 || h != h3)
  {
    std::cout << "size source image and color pass not equal.' " << std::endl;
    hasError = true;
  }
  
  if(w != w4 || h != h4)
  {
    std::cout << "size source image and normal pass not equal.' " << std::endl;
    hasError = true;
  }

  if (hasError)
    return 0;

  uint64_t addressToCkeck = reinterpret_cast<uint64_t>(hdrData.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!

  addressToCkeck = reinterpret_cast<uint64_t>(depth.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!
  
  //
  //
  std::vector<uint32_t> ldrData(hdrData.size());
  const int   windowRadius = 7;
  const int   blockRadius  = 3;
  const float noiseLevel   = 0.1F;  

  bool onGPU = true;

#ifdef MEASURE_TIME // to exclude vulkan initialiяation time from sample execution
  if(onGPU)
  {
    #ifndef NDEBUG
    vk_utils::globalContextInit(std::vector<const char*>(), true);
    #else
    vk_utils::globalContextInit(std::vector<const char*>(), false);
    #endif
  }
#endif

  std::shared_ptr<Denoise> pImpl = nullptr;
  if(onGPU)
    pImpl = CreateDenoise_Generated();
  else
    pImpl = std::make_shared<Denoise>();

  pImpl->PrepareInput(w, h, (const float4*)hdrData.data(), texColor.data(), normal.data(), (const float4*)depth.data());
  pImpl->NLM_denoise (w, h, ldrData.data(), windowRadius, blockRadius, noiseLevel);

  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);  
  else
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);  

  #ifdef MEASURE_TIME
  Denoise_Generated* pGPUImpl = dynamic_cast<Denoise_Generated*>(pImpl.get());
  if(pGPUImpl != nullptr)
  {
    auto timings = pGPUImpl->GetNLM_denoiseExecutionTime();
    std::cout << "NLM_denoise(exec) = " << timings.msExecuteOnGPU                      << " ms " << std::endl;
    std::cout << "NLM_denoise(copy) = " << timings.msCopyToGPU + timings.msCopyFromGPU << " ms " << std::endl;
    std::cout << "NLM_denoise(ovrh) = " << timings.msVulkanInit + timings.msAPIOverhead + timings.msLayoutChange << " ms " << std::endl;
  }
  #endif             

  return 0;
}
