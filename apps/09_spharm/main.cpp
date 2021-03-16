#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <array>

#include "test_class.h"
#include "Bitmap.h"

std::array<LiteMath::float3, 9> process_image_cpu(std::vector<uint32_t>& a_inPixels, uint32_t a_width, uint32_t a_height);
std::array<LiteMath::float3, 9> process_image_gpu(std::vector<uint32_t>& a_inPixels, uint32_t a_width, uint32_t a_height);

int main(int argc, const char** argv)
{
  std::string filename = argv[1];
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP((filename + ".bmp").c_str(), &w, &h);
  
  std::cout << "compute ... " << std::endl;
  auto result  = process_image_cpu(inputImageData, w, h);
  auto result2 = process_image_gpu(inputImageData, w, h);
  
  std::cout << "save to file ... " << std::endl;
  {
    std::ofstream out("out.bin", std::ios::binary);
    for (uint32_t i = 0; i < result.size(); ++i) {
      out.write(reinterpret_cast<char*>(&result[i].x), sizeof(float));
      out.write(reinterpret_cast<char*>(&result[i].y), sizeof(float));
      out.write(reinterpret_cast<char*>(&result[i].z), sizeof(float));
    }
  }
  
  std::cout << std::endl;
  for(size_t i=0;i<result2.size();i++)
  {
    std::cout << result[i].x  << " " << result[i].y  << " " << result[i].z  << "\n" << \
                 result2[i].x << " " << result2[i].y << " " << result2[i].x << " " << std::endl << std::endl;
  }
  
  std::cout << "gen_image ... " << std::endl;
  system(("python3 gen_image.py " + filename).c_str());
  
  return 0;
}
