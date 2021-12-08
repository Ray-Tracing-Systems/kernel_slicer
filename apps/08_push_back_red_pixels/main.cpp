#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"

std::shared_ptr<RedPixels> CreateRedPixels_Generated();

int main(int argc, const char** argv)
{
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP("../01_intersectSphere/zout_cpu.bmp", &w, &h);
  
  bool onGPU = true;

  std::shared_ptr<RedPixels> filter = nullptr;
  
  if(onGPU)
    filter = CreateRedPixels_Generated();
  else
    filter = std::make_shared<RedPixels>();

  filter->SetMaxDataSize(inputImageData.size());
  filter->ProcessPixels(inputImageData.data(), inputImageData.data(), inputImageData.size());

  std::cout << "m_redPixelsNum     = " << filter->m_redPixelsNum << std::endl;
  std::cout << "m_otherPixelsNum   = " << filter->m_otherPixelsNum << std::endl;
  std::cout << "m_testPixelsAmount = " << filter->m_testPixelsAmount << std::endl;
  std::cout << "m_foundPixels_size = " << filter->m_foundPixels.size() << std::endl;
  std::cout << "m_testMin(float)   = " << filter->m_testMin << std::endl;
  std::cout << "m_testMax(float)   = " << filter->m_testMax << std::endl;
  std::cout << "found "                << filter->m_foundPixels.size() << " red pixels" << std::endl;

  if(onGPU)
    SaveBMP("z_out_gpu.bmp", inputImageData.data(), w, h);
  else
    SaveBMP("z_out_cpu.bmp", inputImageData.data(), w, h);
  
  return 0;
}
