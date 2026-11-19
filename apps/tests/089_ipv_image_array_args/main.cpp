#include <iostream>
#include <limits>
#include <memory>
#include <vector>

#include "test_class.h"
#include "ArgParser.h"

#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<ImageArrayArgs> CreateImageArrayArgs_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures ImageArrayArgs_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  ArgParser args(argc, argv);
  const bool onGPU = args.hasOption("--gpu");
  const bool compare = args.hasOption("--compare");
  constexpr int width = 37, height = 19;

  auto run = [&](bool a_gpu)
  {
    std::shared_ptr<ImageArrayArgs> impl;
#ifdef USE_VULKAN
    if(a_gpu)
    {
      auto features = ImageArrayArgs_Generated_ListRequiredDeviceFeatures();
      const auto ctx = vk_utils::globalContextInit(features, args.hasOption("--validation"), args.getOptionValue<int>("--gpu_id", 0));
      impl = CreateImageArrayArgs_Generated(ctx, width*height);
    }
    else
#endif
      impl = std::make_shared<ImageArrayArgs>();

    std::vector<float> output(width*height, std::numeric_limits<float>::quiet_NaN());
    impl->SetSize(width, height);
    impl->CommitDeviceData();
    impl->Run(width, height, output.data());

    impl.reset();
#ifdef USE_VULKAN
    if(a_gpu)
      vk_utils::globalContextDestroy();
#endif

    for(int y = 0; y < height; y++)
      for(int x = 0; x < width; x++)
      {
        const float expected = float(2*((x + 5*y + 2) - (3*x + y + 1)));
        if(output[y*width + x] != expected)
        {
          std::cerr << "Pixel (" << x << ", " << y << "): expected " << expected
                    << ", got " << output[y*width + x] << '\n';
          return false;
        }
      }

    JSONLog::write("pixels", output);
    JSONLog::saveToFile(a_gpu ? "zout_gpu.json" : "zout_cpu.json");
    return true;
  };

#ifndef USE_VULKAN
  if(onGPU || compare)
  {
    std::cerr << "Vulkan support is disabled\n";
    return 1;
  }
#endif

  return (compare ? run(false) && run(true) : run(onGPU)) ? 0 : 1;
}
