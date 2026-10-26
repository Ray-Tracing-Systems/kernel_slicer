#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>
#include <cmath>

#include "test_class.h"
#include "ArgParser.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<TexFilter> CreateTexFilter_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures TexFilter_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const int WIDTH = 256, HEIGHT = 128;
  std::vector<float4> pixels(WIDTH*HEIGHT);
  for(int y = 0; y < HEIGHT; y++)
  {
    for(int x = 0; x < WIDTH; x++)
    {
      const float fx = float(x)/float(WIDTH), fy = float(y)/float(HEIGHT);
      const float stripes = (((x/16) + (y/16)) % 2 == 0) ? 1.0f : 0.25f;
      pixels[y*WIDTH + x] = float4(fx*stripes, fy*stripes, 0.5f*(1.0f + std::sin(fx*12.0f))*stripes, 1.0f);
    }
  }
  Image2D<float4> input(WIDTH, HEIGHT, pixels.data());
  std::vector<uint32_t> out(WIDTH*HEIGHT, 0);

  std::shared_ptr<TexFilter> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = TexFilter_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateTexFilter_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<TexFilter>();

  pImpl->SetSize(WIDTH, HEIGHT);
  pImpl->CommitDeviceData();
  pImpl->Run(WIDTH, HEIGHT, input, out.data());

  std::cout << "out[" << WIDTH*HEIGHT/2 + WIDTH/2 << "] = " << std::hex << out[WIDTH*HEIGHT/2 + WIDTH/2] << std::dec << std::endl;
  LiteImage::SaveBMP(onGPU ? "zout_gpu.bmp" : "zout_cpu.bmp", out.data(), WIDTH, HEIGHT);

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
