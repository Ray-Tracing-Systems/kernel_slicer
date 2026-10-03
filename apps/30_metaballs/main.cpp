#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <cmath>
#include <cstdint>

#include "metaballs.h"
#include "Image2d.h"
#include "ArgParser.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<Metaballs> CreateMetaballs_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures Metaballs_Generated_ListRequiredDeviceFeatures();
#endif

static std::shared_ptr<Metaballs> MakeImpl(bool onGPU, int w, int h, bool enableValidationLayers, int deviceId)
{
  std::shared_ptr<Metaballs> pImpl = nullptr;
  if(onGPU)
  {
    #ifdef USE_VULKAN
    auto features = Metaballs_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, deviceId);
    pImpl         = CreateMetaballs_Generated(ctx, size_t(w)*size_t(h));
    #else
    std::cout << "[main]: built without Vulkan, --gpu is not available" << std::endl;
    return nullptr;
    #endif
  }
  else
    pImpl = std::make_shared<Metaballs>();

  pImpl->SetMaxImageSize(w, h);  // 1) size member vectors
  pImpl->CommitDeviceData();     // 2) upload scene and buffers
  return pImpl;
}

static std::vector<uint32_t> RenderAndReport(std::shared_ptr<Metaballs> pImpl, const char* a_name, int w, int h, int spp)
{
  std::vector<uint32_t> image(size_t(w)*size_t(h));
  pImpl->Render(w, h, spp, image.data());

  float timings[4] = {0,0,0,0};
  pImpl->GetExecutionTime("Render", timings);
  std::cout << "[" << a_name << "] Render(exec) = " << timings[0]              << " ms " << std::endl;
  std::cout << "[" << a_name << "] Render(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
  std::cout << "[" << a_name << "] Render(ovrh) = " << timings[3]              << " ms " << std::endl;
  std::cout << "[" << a_name << "] avg luminance = " << std::exp(pImpl->GetSumLogLum()/float(w*h)) << std::endl;

  const std::string fileName = std::string("zout_") + a_name + ".bmp";
  LiteImage::SaveBMP(fileName.c_str(), image.data(), w, h);
  std::cout << "[" << a_name << "] saved '" << fileName << "'" << std::endl;
  return image;
}

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool compare  = args.hasOption("--compare");
  const bool onGPU    = args.hasOption("--gpu") || compare;
  const bool onCPU    = !args.hasOption("--gpu") || compare;
  const int  deviceId = args.getOptionValue<int>("--gpu_id", 0);
  const int  spp      = args.getOptionValue<int>("--spp",    2);   // spp x spp samples per pixel
  const int  WIDTH    = 1024;
  const int  HEIGHT   = 1024;

  std::vector<uint32_t> cpuImage, gpuImage;
  if(onCPU)
  {
    auto pImpl = MakeImpl(false, WIDTH, HEIGHT, enableValidationLayers, deviceId);
    cpuImage   = RenderAndReport(pImpl, "cpu", WIDTH, HEIGHT, spp);
  }
  if(onGPU)
  {
    auto pImpl = MakeImpl(true, WIDTH, HEIGHT, enableValidationLayers, deviceId);
    if(pImpl == nullptr)
      return -1;
    gpuImage = RenderAndReport(pImpl, "gpu", WIDTH, HEIGHT, spp);
    pImpl    = nullptr;          // destroy generated object before the Vulkan context
    #ifdef USE_VULKAN
    vk_utils::globalContextDestroy();
    #endif
  }

  if(compare)
  {
    // Ray marching makes silhouettes sensitive to the last float bit, so a few pixels may differ.
    size_t badPixels = 0;
    int    maxDiff   = 0;
    for(size_t i = 0; i < cpuImage.size(); i++)
    {
      int pixDiff = 0;
      for(int ch = 0; ch < 3; ch++)
        pixDiff = std::max(pixDiff, std::abs(int((cpuImage[i] >> (8*ch)) & 0xFF) - int((gpuImage[i] >> (8*ch)) & 0xFF)));
      maxDiff = std::max(maxDiff, pixDiff);
      if(pixDiff > 2)
        badPixels++;
    }
    const double badPercent = 100.0*double(badPixels)/double(cpuImage.size());
    const bool   ok         = badPercent < 0.5;
    std::cout << "[compare] pixels differing by more than 2/255: " << badPixels << " (" << badPercent << "%), max diff = " << maxDiff << std::endl;
    std::cout << (ok ? "[compare] CPU and GPU images match" : "[compare] CPU and GPU images DIFFER") << std::endl;
    return ok ? 0 : 1;
  }
  return 0;
}
