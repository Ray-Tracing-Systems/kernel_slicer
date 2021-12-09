#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"

//void tone_mapping_cpu(int w, int h, const float* a_hdrData, const char* a_outName);
//void tone_mapping_gpu(int w, int h, const float* a_hdrData, const char* a_outName);

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp
std::shared_ptr<ToneMapping> CreateToneMapping_Generated(const int w, const int h);

int main(int argc, const char** argv)
{
  std::vector<float> hdrData;
  int w =0, h = 0;

  if(!LoadHDRImageFromFile("../images/nancy_church_2.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'nancy_church_2.hdr' " << std::endl;
    return 0;
  }

  uint64_t addressToCkeck = reinterpret_cast<uint64_t>(hdrData.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!

  //tone_mapping_cpu(w, h, hdrData.data(), "zout_cpu.bmp");
  //tone_mapping_gpu(w, h, hdrData.data(), "zout_gpu.bmp");
  
  bool onGPU = true;
  std::shared_ptr<ToneMapping> pImpl = nullptr;
  if(onGPU)
    pImpl = CreateToneMapping_Generated(w,h);
  else
    pImpl = std::make_shared<ToneMapping>(w,h);
  
  // put data to texture because our class works with textures
  //
  Texture2D<float4> texture(w, h, (const float4*)hdrData.data());
  std::vector<uint> ldrData(w*h);

  pImpl->Bloom(w, h, texture, ldrData.data());
  
  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);  

  return 0;
}
