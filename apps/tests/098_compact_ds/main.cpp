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
std::shared_ptr<TestCompactDS> CreateTestCompactDS_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures TestCompactDS_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t OUT_SIZE = 1000;

  std::shared_ptr<TestCompactDS> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = TestCompactDS_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateTestCompactDS_Generated(ctx, OUT_SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<TestCompactDS>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  // each iteration changes sizes and content of member vectors (or appends to some of them) and commits them again
  //
  for(int iter = 0; iter < 3; iter++)
  {
    std::vector<float> out1(OUT_SIZE), out2(OUT_SIZE);
    pImpl->InitData(iter);
    if(iter == 0)
      pImpl->CommitDeviceData();
    else
      pImpl->CommitAppendedData();
    pImpl->Run(OUT_SIZE, out1.data());
    pImpl->RunRTVBlock(OUT_SIZE, out2.data(), 1);

    std::cout << "iter = " << iter << ", out1[1] = " << out1[1] << ", out2[1] = " << out2[1] << std::endl;
    JSONLog::write(("out1_" + std::to_string(iter)).c_str(), out1);
    JSONLog::write(("out2_" + std::to_string(iter)).c_str(), out2);
  }

  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
