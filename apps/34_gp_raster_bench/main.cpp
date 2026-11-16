#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <string>

#include "Image2d.h"
#include "ArgParser.h"
#include "raster_bench.h"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<RasterBench> CreateRasterBench_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures RasterBench_Generated_ListRequiredDeviceFeatures();
#endif
#ifdef USE_ISPC
std::shared_ptr<RasterBench> CreateRasterBench_ISPC();
#endif

// usage: testapp [--gpu | --ispc] [--bench [--runs N] [--layout linear|tiled|bins] [--binsize 64] [--size 1024] [--algo auto|pixel|block] [--threads 0|1]]
// '--threads 0' disables multithreading of the CPU rasterizer
//
int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  ArgParser args(argc, argv);
  const int WIDTH  = args.hasOption("--size") ? args.getOptionValue<int>("--size") : 1024;
  const int HEIGHT = WIDTH;
  const bool onGPU  = args.hasOption("--gpu");
  const bool onISPC = args.hasOption("--ispc");
  const bool bench  = args.hasOption("--bench");
  const int  runs   = args.hasOption("--runs") ? args.getOptionValue<int>("--runs") : 10;

  LiteImage::Layout layout = LiteImage::Layout::TILED4x4;
  if(args.hasOption("--layout") && args.getOptionValue<std::string>("--layout") == "linear")
    layout = LiteImage::Layout::LINEAR;
  else if(args.hasOption("--layout") && args.getOptionValue<std::string>("--layout") == "bins")
    layout = LiteImage::Layout::TILED4x4_BINS;
  const int binSize = args.hasOption("--binsize") ? args.getOptionValue<int>("--binsize") : 64; // bins of TILED4x4_BINS and of the rasterizer

  std::shared_ptr<RasterBench> pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    auto features = RasterBench_Generated_ListRequiredDeviceFeatures(); // Vulkan 1.3 with '-gp_dynamic_rendering 1'
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, args.getOptionValue<int>("--gpu_id", 0));
    pImpl         = CreateRasterBench_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateRasterBench_ISPC();
  else
  #endif
    pImpl = std::make_shared<RasterBench>();

  pImpl->SetViewport(WIDTH, HEIGHT, layout, binSize);
  LiteRS::Pipeline& pipe = pImpl->GetPipeline();
  pipe.cpu.binSize = binSize;
  if(args.hasOption("--algo"))
  {
    const std::string algo = args.getOptionValue<std::string>("--algo");
    pipe.cpu.algo = (algo == "pixel") ? LiteRS::RasterAlgo::PIXEL : ((algo == "block") ? LiteRS::RasterAlgo::BLOCK : LiteRS::RasterAlgo::AUTO);
  }
  if(args.hasOption("--threads"))
    pipe.cpu.multithreaded = (args.getOptionValue<int>("--threads") != 0);
  pImpl->CommitDeviceData();

  const char* names[3] = {"small", "medium", "large"};
  const char* funcs[3] = {"RenderSmall", "RenderMedium", "RenderLarge"};
  const std::string backend = onGPU ? "gpu" : (onISPC ? "ispc" : "cpu");
  std::vector<uint32_t> pixels(WIDTH*HEIGHT);
  for(int scene = 0; scene < 3; scene++)
  {
    auto render = [&]()
    {
      if(scene == 0)
        pImpl->RenderSmall(WIDTH, HEIGHT, pixels.data());
      else if(scene == 1)
        pImpl->RenderMedium(WIDTH, HEIGHT, pixels.data());
      else
        pImpl->RenderLarge(WIDTH, HEIGHT, pixels.data());
    };

    render(); // warm up and the image for comparison
    LiteImage::Image2D<uint32_t> image(WIDTH, HEIGHT, pixels.data());
    LiteImage::SaveImage(("zout_" + std::string(names[scene]) + "_" + backend + ".bmp").c_str(), image);

    if(!bench)
      continue;
    pipe.stats             = LiteRS::CPUStats();
    pipe.cpu.collectStats  = true;
    double total = 0.0;
    for(int run = 0; run < runs; run++)
    {
      render();
      float timings[4];
      pImpl->GetExecutionTime(funcs[scene], timings);
      total += timings[0];
    }
    pipe.cpu.collectStats = false;
    const double n = double(runs);
    std::cout << std::fixed << std::setprecision(2) << std::setw(6) << names[scene] << ": total " << std::setw(7) << total/n << " ms";
    if(!onGPU)
      std::cout << " | vertex " << std::setw(6) << pipe.stats.vertexMs/n << " | setup " << std::setw(6) << pipe.stats.setupMs/n
                << " | raster " << std::setw(7) << pipe.stats.rasterMs/n << " | triangles " << pipe.stats.triangles/uint64_t(runs);
    std::cout << std::endl;
  }

  pImpl = nullptr;
  #ifdef USE_VULKAN
  if(onGPU)
    vk_utils::globalContextDestroy();
  #endif
  return 0;
}
