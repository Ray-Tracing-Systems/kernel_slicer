#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cassert>

#include "ArgParser.h"
#include "mandelbrot.h"
#include "Image2d.h"

#include "vk_context.h"
std::shared_ptr<Mandelbrot> CreateMandelbrot_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated); 
#ifdef USE_ISPC
std::shared_ptr<Mandelbrot> CreateMandelbrot_ISPC(); 
#endif

int main(int argc, const char** argv)
{
  int w = 1024,h = 1024; 
  std::vector<uint> ldrData(w*h);
  
  ArgParser args(argc, argv);
  bool onGPU  = args.hasOption("--gpu");
  bool isISPC = args.hasOption("--ispc");

  std::shared_ptr<Mandelbrot> pImpl = nullptr;

  if(onGPU)
  {
    auto ctx   = vk_utils::globalContextGet(false, 0);
    pImpl = CreateMandelbrot_Generated(ctx, w*h);
  }
  #ifdef USE_ISPC
  else if(isISPC)
    pImpl = CreateMandelbrot_ISPC();
  #endif
  else
    pImpl = std::make_shared<Mandelbrot>();

  pImpl->CommitDeviceData();
  pImpl->Fractal(w, h, ldrData.data());

  if(onGPU)
    LiteImage::SaveBMP("zout_gpu.bmp", ldrData.data(), w, h);
  else if(isISPC)
    LiteImage::SaveBMP("zout_ispc.bmp", ldrData.data(), w, h);
  else
    LiteImage::SaveBMP("zout_cpu.bmp", ldrData.data(), w, h);

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Fractal", timings);
  std::cout << "Fractal(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "Fractal(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "Fractal(ovrh) = " << timings[3]              << " ms " << std::endl;

  return 0;
}
