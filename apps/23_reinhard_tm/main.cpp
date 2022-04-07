#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "reinhard.h"
#include "Bitmap.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp

int main(int argc, const char** argv)
{
  std::vector<float> hdrData;
  int w,h;
  
  if(!LoadHDRImageFromFile("../images/kitchen.hdr", &w, &h, hdrData))
  {
    std::cout << "can't open input file 'kitchen.hdr' " << std::endl;
    return 0;
  }



  std::vector<uint> ldrData(w*h);

  

  
  SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);
  
  return 0;
}
