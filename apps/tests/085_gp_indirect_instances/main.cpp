#include <iostream>
#include <memory>
#include <vector>

#include "Image2d.h"
#include "ArgParser.h"
#include "test_class.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<IndirectCells> CreateIndirectCells_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures IndirectCells_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<IndirectCells> CreateIndirectCells_ISPC();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const int WIDTH  = IndirectCells::IMAGE_SIZE;
  const int HEIGHT = IndirectCells::IMAGE_SIZE;

  ArgParser args(argc, argv);
  const bool onGPU = args.hasOption("--gpu");
  const bool onISPC = args.hasOption("--ispc");

  std::shared_ptr<IndirectCells> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto features = IndirectCells_Generated_ListRequiredDeviceFeatures(); // Vulkan 1.3 with '-gp_dynamic_rendering 1'
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl         = CreateIndirectCells_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateIndirectCells_ISPC();
  else
  #endif
    pImpl = std::make_shared<IndirectCells>();

  pImpl->SetViewport(WIDTH, HEIGHT);
  pImpl->CommitDeviceData();

  // 4 frames with different numbers of visible cells (instance counts change between frames on GPU) in a 2x2 image
  //
  std::vector<uint32_t> pixels(WIDTH*HEIGHT);
  LiteImage::Image2D<uint32_t> image(2*WIDTH, 2*HEIGHT);
  for(uint32_t frame = 0; frame < 4; frame++)
  {
    pImpl->Render(WIDTH, HEIGHT, frame, pixels.data());
    float timings[4];
    pImpl->GetExecutionTime("Render", timings);
    std::cout << "Render(exec, frame " << frame << ") = " << timings[0] << " ms" << std::endl;
    for(int y = 0; y < HEIGHT; y++)
      for(int x = 0; x < WIDTH; x++)
        image[int2(int(frame % 2)*WIDTH + x, int(frame / 2)*HEIGHT + y)] = pixels[y*WIDTH + x];
  }
  LiteImage::SaveImage(onGPU ? "zout_gpu.bmp" : (onISPC ? "zout_ispc.bmp" : "zout_cpu.bmp"), image);

  pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  return 0;
}
