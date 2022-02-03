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
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#include "vk_context.h"
std::shared_ptr<SphHarm> CreateSphHarm_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);

  std::string filename = args.hasOption("--test") ? "skybox" : argv[1];
  int w, h;
  std::vector<uint32_t> inputImageData = LoadBMP((filename + ".bmp").c_str(), &w, &h);
  if(inputImageData.empty())
    throw std::runtime_error("Failed to load inputImageData from file: " + filename + ".bmp");
  

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
  const int coefsCount = 9;
  std::vector<float> coefs_x(coefsCount);
  std::vector<float> coefs_y(coefsCount);
  std::vector<float> coefs_z(coefsCount);
  for(size_t i=0;i<coefsCount;i++)
  {
    coefs_x[i] = pImpl->coefs[i].x;
    coefs_y[i] = pImpl->coefs[i].y;
    coefs_z[i] = pImpl->coefs[i].z;
  }
  JSONLog::write("coefs_x", coefs_x);
  JSONLog::write("coefs_y", coefs_y);
  JSONLog::write("coefs_z", coefs_z);
  std::string backendName = onGPU ? "gpu" : "cpu";
  JSONLog::saveToFile("zout_"+backendName+".json");
  
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
