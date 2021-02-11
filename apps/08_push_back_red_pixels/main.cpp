#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"

void process_image_cpu(std::vector<uint32_t>& a_inPixels);
void process_image_gpu(std::vector<uint32_t>& a_inPixels);

int main(int argc, const char** argv)
{
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP("../01_intersectSphere/zout_cpu.bmp", &w, &h);
  
  std::vector<uint32_t> resCPU = inputImageData; 
  std::vector<uint32_t> resGPU = inputImageData;
  process_image_cpu(resCPU);
  process_image_gpu(resGPU);

  std::cout << "found " << resCPU.size() << " red pixels" << std::endl;
  
  //SaveBMP("z_out_cpu.bmp", resCPU.data(), w, h);
  //SaveBMP("z_out_gpu.bmp", resGPU.data(), w, h);
  
  return 0;
}
