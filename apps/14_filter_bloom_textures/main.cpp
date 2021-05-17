#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "texture2d.h"

void tone_mapping_cpu(int w, int h, const float* a_hdrData, const char* a_outName);
void tone_mapping_gpu(int w, int h, const float* a_hdrData, const char* a_outName);

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp

int main(int argc, const char** argv)
{
  std::vector<float> hdrData;
  int w =0, h = 0;

  if(!LoadHDRImageFromFile("../05_filter_bloom_good/nancy_church_2.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'nancy_church_2.hdr' " << std::endl;
    return 0;
  }

  uint64_t addressToCkeck = reinterpret_cast<uint64_t>(hdrData.data());
  assert(addressToCkeck % 16 == 0); // check if address is aligned!!!

  tone_mapping_cpu(w, h, hdrData.data(), "zout_cpu.bmp");
  //tone_mapping_gpu(w, h, hdrData.data(), "zout_gpu.bmp");
  return 0;
}
