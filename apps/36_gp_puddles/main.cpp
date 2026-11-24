#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "Image2d.h"
#include "ArgParser.h"
#include "puddles.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<PuddleRender> CreatePuddleRender_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures PuddleRender_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<PuddleRender> CreatePuddleRender_ISPC();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const bool onGPU  = args.hasOption("--gpu");
  const bool onISPC = args.hasOption("--ispc");
  const bool isTest = args.hasOption("--test");
  const int  WIDTH  = args.getOptionValue<int>("--width",  isTest ? 512 : 1280);
  const int  HEIGHT = args.getOptionValue<int>("--height", isTest ? 512 : 720);
  const std::string scenePath = args.getOptionValue<std::string>("--scene", "../resources/msu-graphics-group/scenes/01_simple_scenes/instanced_objects.xml");

  std::shared_ptr<PuddleRender> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto features = PuddleRender_Generated_ListRequiredDeviceFeatures(); // ray queries; Vulkan 1.3 with '-gp_dynamic_rendering 1'
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl         = CreatePuddleRender_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreatePuddleRender_ISPC();
  else
  #endif
    pImpl = std::make_shared<PuddleRender>();

  if(!pImpl->LoadScene(scenePath.c_str()))
    return -1;
  pImpl->SetViewport(WIDTH, HEIGHT);
  pImpl->CommitDeviceData();

  std::vector<uint32_t> pixels(WIDTH*HEIGHT);
  pImpl->Render(WIDTH, HEIGHT, pixels.data());

  float timings[4];
  pImpl->GetExecutionTime("Render", timings);
  std::cout << "Render(exec) = " << timings[0] << " ms" << std::endl;

  LiteImage::Image2D<uint32_t> image(WIDTH, HEIGHT, pixels.data());
  LiteImage::SaveImage(onGPU ? "zout_gpu.bmp" : (onISPC ? "zout_ispc.bmp" : "zout_cpu.bmp"), image);

  pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  return 0;
}
