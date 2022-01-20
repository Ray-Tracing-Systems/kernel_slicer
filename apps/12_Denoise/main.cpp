#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <memory>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data);   // defined in imageutils.cpp
bool LoadLDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<int32_t>& a_data); // defined in imageutils.cpp

#include "vk_context.h"
std::shared_ptr<Denoise> CreateDenoise_Generated(const int w, const int h, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

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
    return -1;


  uint64_t addressToCkeck = reinterpret_cast<uint64_t>(hdrData.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!

  addressToCkeck = reinterpret_cast<uint64_t>(depth.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!
  
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const int   windowRadius = 7;
  const int   blockRadius  = 3;
  const float noiseLevel   = 0.1F;
  
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  std::shared_ptr<Denoise> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateDenoise_Generated(w, h, ctx, w*h);
  }
  else
    pImpl = std::make_shared<Denoise>(w,h);
  
  pImpl->CommitDeviceData();

  std::vector<uint> ldrData(w*h);
  pImpl->NLM_denoise(w, h, (const float4*)hdrData.data(), ldrData.data(), texColor.data(), normal.data(), (const float4*)depth.data(), windowRadius, blockRadius, noiseLevel);

  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else 
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("NLM_denoise", timings);
  std::cout << "NLM_denoise(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "NLM_denoise(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "NLM_denoise(ovrh) = " << timings[3]              << " ms " << std::endl;            
  return 0;
}
