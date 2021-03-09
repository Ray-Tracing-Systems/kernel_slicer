#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>

#include "test_class.h"
#include "Bitmap.h"

std::array<LiteMath::float3, 9> process_image_cpu(std::vector<uint32_t>& a_inPixels, uint32_t a_width, uint32_t a_height);
// void process_image_gpu(std::vector<uint32_t>& a_inPixels);

int main(int argc, const char** argv)
{
  std::string filename = argv[1];
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP((filename + ".bmp").c_str(), &w, &h);
  
  std::vector<uint32_t> resCPU = inputImageData; 
  std::vector<uint32_t> resGPU = inputImageData;
  auto result = process_image_cpu(resCPU, w, h);
  {
    std::ofstream out("out.bin", std::ios::binary);
    for (uint32_t i = 0; i < result.size(); ++i) {
      out.write(reinterpret_cast<char*>(&result[i]), sizeof(result[i]));
    }
  }
  system(("python3 gen_image.py " + filename).c_str());
  
  return 0;
}
