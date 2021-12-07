#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"

void tone_mapping_cpu(int w, int h, float* a_hdrData, const char* a_outName);
void tone_mapping_gpu(int w, int h, float* a_hdrData, const char* a_outName);

bool LoadHDRImageFromFile(const char* a_fileName, 
                          int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp

ToneMapping* CreateToneMapping_Generated();

int main(int argc, const char** argv)
{
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
  
  ToneMapping* pImpl = nullptr;
  bool onGPU = true;
  
  if(onGPU)
    pImpl = CreateToneMapping_Generated();
  else
    pImpl = new ToneMapping;

  pImpl->SetMaxImageSize(w,h);
  pImpl->Bloom(w, h, (const LiteMath::float4*)hdrData.data(), ldrData.data());

  if(onGPU)
    SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else
    SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);

  delete pImpl;  
  return 0;
}
