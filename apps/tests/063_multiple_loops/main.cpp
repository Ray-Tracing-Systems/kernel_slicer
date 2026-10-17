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
std::shared_ptr<MultiLoops> CreateMultiLoops_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures MultiLoops_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000, WIDTH = 67, HEIGHT = 45;
  std::vector<float> in(SIZE), out(SIZE, 0.0f), img(WIDTH*HEIGHT), blurred(WIDTH*HEIGHT, 0.0f);
  for(uint32_t i=0;i<SIZE;i++)
    in[i] = float(i % 17)*0.25f;
  for(uint32_t i=0;i<img.size();i++)
    img[i] = float((i*7) % 13);

  std::shared_ptr<MultiLoops> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = MultiLoops_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateMultiLoops_Generated(ctx, WIDTH*HEIGHT);
  }
  else
  #endif
    pImpl = std::make_shared<MultiLoops>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetMaxSize(WIDTH*HEIGHT);
  pImpl->CommitDeviceData();
  pImpl->Run(in.data(), out.data(), SIZE);
  pImpl->Blur(img.data(), blurred.data(), WIDTH, HEIGHT);

  std::cout << "out[100] = " << out[100] << ", m_summ = " << pImpl->m_summ << ", m_result = " << pImpl->m_result << ", m_count = " << pImpl->m_count << ", blurred[500] = " << blurred[500] << std::endl;

  JSONLog::write("out", out);
  JSONLog::write("blurred", blurred);
  JSONLog::write("summ", pImpl->m_summ);
  JSONLog::write("result", pImpl->m_result);
  JSONLog::write("count", pImpl->m_count);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
