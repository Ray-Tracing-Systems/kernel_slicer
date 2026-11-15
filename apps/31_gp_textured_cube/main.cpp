#include <iostream>
#include <memory>
#include <vector>

#include "Image2d.h"
#include "ArgParser.h"
#include "textured_cube.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<TexturedCube> CreateTexturedCube_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures TexturedCube_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const int WIDTH  = 512;
  const int HEIGHT = 512;

  ArgParser args(argc, argv);
  const bool onGPU = args.hasOption("--gpu");

  std::shared_ptr<TexturedCube> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto features = TexturedCube_Generated_ListRequiredDeviceFeatures(); // Vulkan 1.3 with '-gp_dynamic_rendering 1'
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl         = CreateTexturedCube_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<TexturedCube>();

  pImpl->SetViewport(WIDTH, HEIGHT);
  pImpl->CommitDeviceData();

  std::vector<uint32_t> pixels(WIDTH*HEIGHT);
  pImpl->Render(WIDTH, HEIGHT, 0.6f, pixels.data());

  float timings[4];
  pImpl->GetExecutionTime("Render", timings);
  std::cout << "Render(exec) = " << timings[0] << " ms" << std::endl;

  LiteImage::Image2D<uint32_t> image(WIDTH, HEIGHT, pixels.data());
  LiteImage::SaveImage(onGPU ? "zout_gpu.bmp" : "zout_cpu.bmp", image);

  pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  return 0;
}
