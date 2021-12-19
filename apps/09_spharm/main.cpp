#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <array>
#include <memory>

#include "test_class.h"
#include "Bitmap.h"
#include "ArgParser.h"

#include "vk_context.h"
std::shared_ptr<SphHarm> CreateSphHarm_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::string filename = argv[1];
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP((filename + ".bmp").c_str(), &w, &h);
  
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  std::shared_ptr<SphHarm> pImpl = nullptr;
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateSphHarm_Generated(ctx, inputImageData.size());
  }
  else
    pImpl = std::make_shared<SphHarm>();

  pImpl->CommitDeviceData();

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

  std::cout << std::endl;  
  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("ProcessPixels", timings);
  std::cout << "ProcessPixels(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "ProcessPixels(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "ProcessPixels(ovrh) = " << timings[3]              << " ms " << std::endl;
  return 0;
}
