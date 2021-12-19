#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<ToneMapping> CreateToneMapping_Generated(const int w, const int h, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  
  std::vector<float> hdrData;
  int w =0, h = 0;

  if(!LoadHDRImageFromFile("../images/nancy_church_2.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'nancy_church_2.hdr' " << std::endl;
    return 0;
  }

  uint64_t addressToCkeck = reinterpret_cast<uint64_t>(hdrData.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!
  
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  std::shared_ptr<ToneMapping> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateToneMapping_Generated(w,h,ctx,w*h);
  }
  else
    pImpl = std::make_shared<ToneMapping>(w,h);
  
  pImpl->CommitDeviceData();

  // put data to texture because our class works with textures
  //
  Texture2D<float4> texture(w, h, (const float4*)hdrData.data());
  std::vector<uint> ldrData(w*h);

  pImpl->Bloom(w, h, texture, ldrData.data());
  
  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);  

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Bloom", timings);
  std::cout << "Bloom(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "Bloom(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "Bloom(ovrh) = " << timings[3]              << " ms " << std::endl;
  return 0;
}
