#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<WgSizeAll> CreateWgSizeAll_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures WgSizeAll_Generated_ListRequiredDeviceFeatures();
#endif

/// 1 if the generated shader (Slang or GLSL) of a kernel has the work group size of kmake.json
static float ShaderHasWorkGroupSize(const std::string& a_kernel, int x, int y)
{
  for(const char* ext : {".slang", ".comp"})
  {
    std::ifstream fin("shaders_generated/" + a_kernel + ext);
    if(!fin.is_open())
      continue;
    std::stringstream text;
    text << fin.rdbuf();
    const std::string slang = "numthreads(" + std::to_string(x) + ", " + std::to_string(y) + ", 1)";
    const std::string glsl  = "local_size_x = " + std::to_string(x) + ", local_size_y = " + std::to_string(y) + ",";
    const bool found = (text.str().find(slang) != std::string::npos) || (text.str().find(glsl) != std::string::npos);
    std::cout << a_kernel << ext << ": work group " << x << "x" << y << (found ? " found" : " NOT found") << std::endl;
    return found ? 1.0f : 0.0f;
  }
  std::cout << "shader of " << a_kernel << " is not found" << std::endl;
  return 0.0f;
}

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000;
  std::vector<float> in(SIZE), out(SIZE, 0.0f);
  for(uint32_t i = 0; i < SIZE; i++)
    in[i] = float(i % 23) - 11.0f;

  std::shared_ptr<WgSizeAll> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = WgSizeAll_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateWgSizeAll_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<WgSizeAll>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->Run(in.data(), out.data(), SIZE);

  // CPU writes the expected values, GPU checks the generated shaders
  std::vector<float> wgSizes = {1.0f, 1.0f};
  if(onGPU)
    wgSizes = { ShaderHasWorkGroupSize("kernel1D_Scale", 64, 1), ShaderHasWorkGroupSize("kernel2D_Fill", 8, 4) };

  JSONLog::write("out", out);
  JSONLog::write("wgSizeFromConfig", wgSizes);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
