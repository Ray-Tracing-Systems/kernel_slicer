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
vk_utils::VulkanDeviceFeatures Numbers_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<float> A(1024), B(1024), C(1024), D(1024);
  for(size_t i=0;i<A.size();i++)
  {
    A[i] = float(i);
    B[i] = float(i*2);
    C[i] = float(i*3);
    D[i] = -float(i);
  }

  std::shared_ptr<Numbers> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = Numbers_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateNumbers_Generated(ctx, A.size());
  }
  else
  #endif
    pImpl = std::make_shared<Numbers>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->CommitDeviceData();
  pImpl->SAXPY(A.data(), B.data(), C.data(), D.data(), unsigned(A.size()));

  for(int i=0;i<10;i++)
    std::cout << i << "\t" << D[i] << "\t" << i*(2*i) + 3*i + 3 << std::endl;

  //JSONLog::write("array summ", pImpl->m_summ);
  //JSONLog::saveToFile("zout_"+backendName+".json");
  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
