#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <array>
#include <memory>

#include "test_class.h"
#include "Bitmap.h"

std::shared_ptr<SphHarm> CreateSphHarm_Generated();

int main(int argc, const char** argv)
{
  std::string filename = argv[1];
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP((filename + ".bmp").c_str(), &w, &h);
  
  bool onGPU = true;
  std::shared_ptr<SphHarm> pImpl = nullptr;
  if(onGPU)
    pImpl = CreateSphHarm_Generated();
  else
    pImpl = std::make_shared<SphHarm>();

  std::cout << "compute ... " << std::endl;
  pImpl->ProcessPixels(inputImageData.data(), w, h);
  
  std::cout << "save to file ... " << std::endl;
  {
    std::ofstream out("out.bin", std::ios::binary);
    for (uint32_t i = 0; i < 9; ++i) {
      out.write(reinterpret_cast<char*>(&pImpl->coefs[i].x), sizeof(float));
      out.write(reinterpret_cast<char*>(&pImpl->coefs[i].y), sizeof(float));
      out.write(reinterpret_cast<char*>(&pImpl->coefs[i].z), sizeof(float));
    }
  }
  
  std::cout << std::endl;
  for(size_t i=0;i<9;i++)
    std::cout << pImpl->coefs[i].x  << " " << pImpl->coefs[i].y  << " " << pImpl->coefs[i].z  << std::endl;
  
  std::cout << "gen_image ... " << std::endl;
  system(("python3 gen_image.py " + filename).c_str());
  
  return 0;
}
