#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "reinhard.h"

bool LoadHDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<float>& a_data); // defined in imageutils.cpp
void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);

float reinhard_extended(float v, float max_white)
{
  float numerator = v * (1.0f + (v / float(max_white * max_white)));
  return numerator / (1.0f + v);
}


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

  // Reinhard tm algorithm
  //
  float whitePoint = 0.0f;

  for(int i=0;i<w*h;i++)
  {
    float maxColor = std::max(hdrData[i*4+0], std::max(hdrData[i*4+1], hdrData[i*4+2]));
    whitePoint = std::max(whitePoint, maxColor);
  }

  for(int y=0;y<h;y++)
  {
    for(int x=0;x<w;x++)
    {
      int offset  = (y*w+x)*4;
      float red   = reinhard_extended(hdrData[offset+0], whitePoint);
      float green = reinhard_extended(hdrData[offset+1], whitePoint);
      float blue  = reinhard_extended(hdrData[offset+2], whitePoint);

      uint32_t r = std::min(red*255.0f,   255.0f);
      uint32_t g = std::min(green*255.0f, 255.0f);
      uint32_t b = std::min(blue*255.0f,  255.0f);

      ldrData[y*w+x] = 0xFF000000 | r | (g << 8) | (b << 16);
    }
  }

  
  SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);
  
  return 0;
}
