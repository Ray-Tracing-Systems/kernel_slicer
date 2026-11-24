#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

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
  const int  RUNS   = std::max(args.getOptionValue<int>("--runs", 1), 1);
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

  // the first run is a warm up when several runs are requested; time is the wall clock time of a control function call,
  // for GPU it includes submit and wait (and copying of the image back for 'ShadeAndReflect')
  //
  const bool stats = args.hasOption("--stats"); // time of stages of rasterization on CPU (C++ and ISPC)
  pImpl->SetCollectStats(stats);

  std::vector<uint32_t> pixels(WIDTH*HEIGHT);
  float timeRaster = 0.0f, timeRT = 0.0f, gpuRaster = 0.0f, gpuRT = 0.0f;
  const int warmUp = (RUNS > 1) ? 1 : 0;
  for(int run = 0; run < RUNS + warmUp; run++)
  {
    auto t0 = std::chrono::high_resolution_clock::now();
    pImpl->RenderGBuffer(WIDTH, HEIGHT);
    auto t1 = std::chrono::high_resolution_clock::now();
    pImpl->ShadeAndReflect(WIDTH, HEIGHT, pixels.data());
    auto t2 = std::chrono::high_resolution_clock::now();
    if(run < warmUp)
    {
      pImpl->GetRasterStats(true);
      continue;
    }
    float timings[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    pImpl->GetExecutionTime("RenderGBuffer", timings);
    gpuRaster += timings[0];
    pImpl->GetExecutionTime("ShadeAndReflect", timings);
    gpuRT += timings[0];
    timeRaster += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()/1000.f;
    timeRT     += std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()/1000.f;
  }

  std::cout << "resolution      = " << WIDTH << "x" << HEIGHT << ", runs = " << RUNS << std::endl;
  std::cout << "RenderGBuffer   = " << timeRaster/float(RUNS) << " ms (GetExecutionTime: " << gpuRaster/float(RUNS) << " ms)" << std::endl;
  std::cout << "ShadeAndReflect = " << timeRT/float(RUNS)     << " ms (GetExecutionTime: " << gpuRT/float(RUNS)     << " ms)" << std::endl;
  if(stats)
  {
    const LiteRS::CPUStats st = pImpl->GetRasterStats(true);
    std::cout << "  vertex shader = " << st.vertexMs/double(RUNS) << " ms" << std::endl;
    std::cout << "  setup, binning= " << st.setupMs /double(RUNS) << " ms" << std::endl;
    std::cout << "  raster and PS = " << st.rasterMs/double(RUNS) << " ms" << std::endl;
    std::cout << "  triangles     = " << st.triangles/uint64_t(RUNS) << " after setup" << std::endl;
  }

  LiteImage::Image2D<uint32_t> image(WIDTH, HEIGHT, pixels.data());
  LiteImage::SaveImage(onGPU ? "zout_gpu.bmp" : (onISPC ? "zout_ispc.bmp" : "zout_cpu.bmp"), image);

  pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  return 0;
}
