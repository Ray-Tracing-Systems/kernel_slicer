#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"

void process_image_cpu(const std::vector<uint32_t>& a_inPixels, std::vector<RedPixels::PixelInfo>& a_outPixels);
void process_image_gpu(const std::vector<uint32_t>& a_inPixels, std::vector<RedPixels::PixelInfo>& a_outPixels);

int main(int argc, const char** argv)
{
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP("../01_intersectSphere/zout_cpu.bmp", &w, &h);
  
  std::vector<RedPixels::PixelInfo> resCPU, resGPU;
  process_image_cpu(inputImageData, resCPU);
  process_image_gpu(inputImageData, resGPU);

  std::cout << "found " << resCPU.size() << " red pixels" << std::endl;
  
  
  return 0;
}
