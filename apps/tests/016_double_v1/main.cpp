#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <cstdint>

#include "test_class.h"
#include "ArgParser.h"
#define JSON_LOG_IMPLEMENTATION
#include "JSONLog.hpp"

#ifdef USE_VULKAN
#include "vk_context.h"
std::shared_ptr<Numbers> CreateNumbers_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
#endif
#ifdef USE_ISPC
std::shared_ptr<Numbers> CreateNumbers_ISPC();
#endif

int32_t array_summ_cpu(const std::vector<int32_t>& array);
int32_t array_summ_gpu(const std::vector<int32_t>& array);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<double> array(1024);
  for(int i=0;i<array.size();i++)
  {
    if(i%3 == 0)
      array[i] = double(i);
    else
      array[i] = double(-i);
  }

  std::shared_ptr<Numbers> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  bool onISPC = args.hasOption("--ispc");
  
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateNumbers_Generated(ctx, array.size());
  }
  else
  #endif
  #ifdef USE_ISPC
  if(onISPC)
    pImpl = CreateNumbers_ISPC();
  else
  #endif
    pImpl = std::make_shared<Numbers>();

  std::string backendName = onGPU ? "gpu" : (onISPC ? "ispc" : "cpu");

  pImpl->CommitDeviceData();
  pImpl->CalcArraySumm(array.data(), unsigned(array.size()));

  JSONLog::write("array summ", pImpl->m_summ);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;           // Vulkan objects of the class first, then the context
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
