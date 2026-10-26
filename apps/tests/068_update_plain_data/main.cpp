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
std::shared_ptr<PlainUpdate> CreatePlainUpdate_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures PlainUpdate_Generated_ListRequiredDeviceFeatures();
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  const uint32_t SIZE = 1000;
  std::vector<float> in(SIZE), table(37), out1(SIZE, 0.0f), out2(SIZE, 0.0f), out3(SIZE, 0.0f);
  for(uint32_t i = 0; i < SIZE; i++)
    in[i] = float(i % 23) - 11.0f;
  for(uint32_t i = 0; i < table.size(); i++)
    table[i] = float(i)*0.125f;

  std::shared_ptr<PlainUpdate> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = PlainUpdate_Generated_ListRequiredDeviceFeatures();
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreatePlainUpdate_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<PlainUpdate>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetTable(table);
  pImpl->SetParameters(1.0f, 0.0f, 0);
  pImpl->CommitDeviceData();
  pImpl->Run(in.data(), out1.data(), SIZE);

  pImpl->SetParameters(2.0f, 0.5f, 5);    // change plain members only
  pImpl->UpdateMembersPlainData();
  pImpl->Run(in.data(), out2.data(), SIZE);

  pImpl->SetParameters(-1.0f, 3.0f, 100);
  pImpl->UpdateMembersPlainData();
  pImpl->Run(in.data(), out3.data(), SIZE);

  std::cout << "out1[10] = " << out1[10] << ", out2[10] = " << out2[10] << ", out3[10] = " << out3[10] << std::endl;

  JSONLog::write("out1", out1);
  JSONLog::write("out2", out2);
  JSONLog::write("out3", out3);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
