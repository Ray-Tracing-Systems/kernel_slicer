#include <iostream>
#include <memory>
#include <vector>

#include "Image2d.h"
#include "ArgParser.h"
#include "test_class.h"

#ifdef USE_VULKAN
#include "vk_context.h"
#include "test_class_generated.h"
std::shared_ptr<TriangleRender> CreateTriangleRender_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures TriangleRender_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<TriangleRender> CreateTriangleRender_ISPC();
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
  const bool onISPC = args.hasOption("--ispc");

  std::shared_ptr<TriangleRender> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto features = TriangleRender_Generated_ListRequiredDeviceFeatures(); // Vulkan 1.3 with '-gp_dynamic_rendering 1'
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl         = CreateTriangleRender_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateTriangleRender_ISPC();
  else
  #endif
    pImpl = std::make_shared<TriangleRender>();

  pImpl->SetViewport(WIDTH, HEIGHT);
  pImpl->CommitDeviceData();

  std::vector<uint32_t> pixels(WIDTH*HEIGHT);
  pImpl->Render(WIDTH, HEIGHT, 0.25f, pixels.data());

  float timings[4];
  pImpl->GetExecutionTime("Render", timings);
  std::cout << "Render(exec) = " << timings[0] << " ms" << std::endl;

  LiteImage::Image2D<uint32_t> image(WIDTH, HEIGHT, pixels.data());
  LiteImage::SaveImage(onGPU ? "zout_gpu.bmp" : (onISPC ? "zout_ispc.bmp" : "zout_cpu.bmp"), image);

  #ifdef USE_VULKAN
  if(onGPU)
  {
    // Resize both ways without re-committing geometry, then check a full commit too.
    for(int pass=0; pass<3; pass++)
    {
      const int w = pass == 0 ? 768 : 300;
      const int h = pass == 0 ? 384 : 200;
      pImpl->SetViewport(w,h);
      if(pass < 2)
        std::dynamic_pointer_cast<TriangleRender_Generated>(pImpl)->ResizeGraphicsTargets();
      else
        pImpl->CommitDeviceData();
      TriangleRender reference;
      reference.SetViewport(w,h);
      std::vector<uint32_t> expected(w*h), actual(w*h);
      reference.Render(w,h,0.25f,expected.data());
      pImpl->Render(w,h,0.25f,actual.data());
      size_t errors = 0;
      for(size_t i=0;i<expected.size();i++)
        for(int shift=0;shift<24;shift+=8)
          if(std::abs(int((expected[i]>>shift)&255)-int((actual[i]>>shift)&255)) > 3)
          { errors++; break; }
      if(errors > expected.size()/100)
      {
        std::cerr << "resize pass " << pass << ": " << errors << " mismatches" << std::endl;
        return 1;
      }
    }
  }
  #endif

  pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  return 0;
}
