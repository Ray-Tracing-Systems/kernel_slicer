#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>
#include <cmath>
#include <algorithm>

#include "test_class.h"
#include "ArgParser.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data);   // defined in imageutils.cpp
bool LoadLDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<int32_t>& a_data); // defined in imageutils.cpp

#include "vk_context.h"
std::shared_ptr<Denoise> CreateDenoise_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures Denoise_Generated_ListRequiredDeviceFeatures();

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

  //
  //
  std::vector<uint32_t> ldrData(hdrData.size());
  const int   windowRadius = 7;
  const int   blockRadius  = 3;
  const float noiseLevel   = 0.1F;  

  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<Denoise> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = Denoise_Generated_ListRequiredDeviceFeatures();
    auto ctx = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateDenoise_Generated(ctx, w*h);
  }
  else
    pImpl = std::make_shared<Denoise>();
  
  pImpl->PrepareInput(w, h, (const float4*)hdrData.data(), texColor.data(), normal.data(), (const float4*)depth.data());
  pImpl->CommitDeviceData();
  
  // Keep each GPU submission short enough for the driver's hang timeout.
  const int rowsPerSubmission = onGPU ? 64 : h;
  float timings[4] = {};
  for(int firstRow = 0; firstRow < h; firstRow += rowsPerSubmission)
  {
    const int rowCount = std::min(rowsPerSubmission, h - firstRow);
    pImpl->NLM_denoise(w, rowCount, ldrData.data() + size_t(firstRow)*w, windowRadius, blockRadius, noiseLevel, firstRow);
    float bandTimings[4] = {};
    pImpl->GetExecutionTime("NLM_denoise", bandTimings);
    for(int i = 0; i < 4; ++i)
      timings[i] += bandTimings[i];
  }

  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);  
  else
    LiteImage::SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);  

  std::cout << "NLM_denoise(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "NLM_denoise(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "NLM_denoise(ovrh) = " << timings[3]              << " ms " << std::endl;
  pImpl.reset();
  if(onGPU)
    vk_utils::globalContextDestroy();
  return 0;
}
