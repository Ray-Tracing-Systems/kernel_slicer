#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Image2d.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<ToneMapping> CreateToneMapping_GPU(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
#ifdef USE_ISPC
std::shared_ptr<ToneMapping> CreateToneMapping_ISPC();
#endif

bool LoadHDRImageFromFile(const char* a_fileName, 
                          int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> hdrData;
  int w,h;
  if(!LoadHDRImageFromFile("../images/nancy_church_2.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'nancy_church_2.hdr' " << std::endl;
    return 0;
  }
  std::vector<uint> ldrData(w*h);

  uint64_t addressToCkeck = reinterpret_cast<uint64_t>(hdrData.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!
  
  std::shared_ptr<ToneMapping> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU  = args.hasOption("--gpu");
  bool isISPC = args.hasOption("--ispc");
  
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateToneMapping_GPU(ctx, w*h);
  }
  #ifdef USE_ISPC
  else if(isISPC)
    pImpl = CreateToneMapping_ISPC();
  #endif
  else
    pImpl = std::make_shared<ToneMapping>();

  pImpl->SetMaxImageSize(w,h);
  pImpl->CommitDeviceData();

  pImpl->Bloom(w, h, (const LiteMath::float4*)hdrData.data(), ldrData.data());

  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else if(isISPC)
    LiteImage::SaveBMP("zout_ispc.bmp", ldrData.data(), w, h);
  else
    LiteImage::SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);
  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Bloom", timings);
  std::cout << "Bloom(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "Bloom(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "Bloom(ovrh) = " << timings[3]              << " ms " << std::endl;

  pImpl->GetExecutionTime("kernel2D_BlurX", timings);
  std::cout << "kernel2D_BlurX(avg) = " << timings[0] << " ms " << std::endl;
  std::cout << "kernel2D_BlurX(min) = " << timings[1] << " ms " << std::endl;
  std::cout << "kernel2D_BlurX(max) = " << timings[2] << " ms " << std::endl;

  pImpl = nullptr;
  vk_utils::globalContextDestroy();  
  return 0;
}
