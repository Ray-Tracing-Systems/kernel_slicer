#include <iostream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<KernelLocals> CreateKernelLocals_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures KernelLocals_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000, WIDTH = 64, HEIGHT = 48;
  std::vector<float> in(SIZE), out(SIZE), out2(WIDTH*HEIGHT);
  for(uint32_t i=0;i<SIZE;i++)
    in[i] = float(i)/float(SIZE);

  std::shared_ptr<KernelLocals> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = KernelLocals_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateKernelLocals_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<KernelLocals>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetWeights({0.25f, 0.75f, 3.0f});
  pImpl->CommitDeviceData();
  pImpl->Run(in.data(), out.data(), SIZE, 2.2f);
  pImpl->Run2D(out2.data(), WIDTH, HEIGHT);

  std::cout << "out[500] = " << out[500] << ", m_result = " << pImpl->m_result << ", out2[100] = " << out2[100] << std::endl;

  JSONLog::write("out", out);
  JSONLog::write("out2", out2);
  JSONLog::write("result", pImpl->m_result);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
