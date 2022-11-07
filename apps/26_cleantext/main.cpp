#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "removal.h"

bool LoadLDRImageFromFile(const char* a_fileName, int* pW, int* pH, std::vector<int32_t>& a_data);
void SaveBMP(const char* fname, const unsigned int* pixels, int w, int h);

#include "vk_context.h"

int main(int argc, const char** argv)
{
  std::vector<int> imageData;
  int w,h;
  
  if(!LoadLDRImageFromFile("valka.jpg", &w, &h, imageData))
  {
    std::cout << "can't open input file 'valka.jpg' " << std::endl;
    return 0;
  }

  bool onGPU  = false;  // args.hasOption("--gpu");
  bool isISPC = false; // args.hasOption("--ispc");

  std::vector<uint32_t> outData(w*h);

  //if(onGPU)
  //{
  //  auto ctx   = vk_utils::globalContextGet(false, 0);
  //  pImpl = CreateReinhardTM_Generated(ctx, w*h);
  //}
  //else if(isISPC)
  //{
  //  pImpl = CreateReinhardTM_ISPC();
  //}
  //else
  auto pImpl = std::make_shared<TextRemoval>();

  pImpl->Reserve(w,h);
  pImpl->CommitDeviceData();
  pImpl->Run(w, h, (const uint32_t*)imageData.data(), outData.data());

  if(onGPU)
    SaveBMP("valka_gpu.bmp", outData.data(), w, h);
  else if(isISPC)
    SaveBMP("valka_ispc.bmp", outData.data(), w, h);
  else
    SaveBMP("valka_cpu.bmp", outData.data(), w, h);

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Run", timings);
  std::cout << "Run(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "Run(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "Run(ovrh) = " << timings[3]              << " ms " << std::endl;

  return 0;
}
