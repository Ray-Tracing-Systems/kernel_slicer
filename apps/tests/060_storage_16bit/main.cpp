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
std::shared_ptr<Storage16> CreateStorage16_Generated(vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);
vk_utils::VulkanDeviceFeatures Storage16_Generated_ListRequiredDeviceFeatures();

static bool Has16BitStorageRequest(const vk_utils::VulkanDeviceFeatures& a_features)
{
  const VkBaseInStructure* pNext = (const VkBaseInStructure*)a_features.features2.pNext;
  while(pNext != nullptr)
  {
    if(pNext->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES)
      return ((const VkPhysicalDevice16BitStorageFeatures*)pNext)->storageBuffer16BitAccess == VK_TRUE;
    pNext = pNext->pNext;
  }
  return false;
}
#endif

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif
  const uint32_t SIZE = 1000;
  std::vector<PackedVertex> verts(SIZE);
  for(uint32_t i=0;i<SIZE;i++)
  {
    verts[i].u        = uint16_t((i*977u) % 65536u);
    verts[i].v        = uint16_t(i % 300u);
    verts[i].id       = i;
    verts[i].flags[0] = uint16_t(i*3);
    verts[i].flags[1] = uint16_t(65535u - i);
  }
  std::vector<float> out(SIZE);

  std::shared_ptr<Storage16> pImpl = nullptr;
  ArgParser args(argc, argv);

  bool onGPU = args.hasOption("--gpu");
  if(args.hasOption("--validation"))
    enableValidationLayers = true;
  #ifdef USE_VULKAN
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto features = Storage16_Generated_ListRequiredDeviceFeatures();
    if(!Has16BitStorageRequest(features))
    {
      std::cout << "FAILED: storageBuffer16BitAccess is not requested in ListRequiredDeviceFeatures" << std::endl;
      return 1;
    }
    auto ctx      = vk_utils::globalContextInit(features, enableValidationLayers, a_preferredDeviceId);
    pImpl         = CreateStorage16_Generated(ctx, SIZE);
  }
  else
  #endif
    pImpl = std::make_shared<Storage16>();

  std::string backendName = onGPU ? "gpu" : "cpu";

  pImpl->SetData(verts);
  pImpl->CommitDeviceData();
  pImpl->Run(out.data(), SIZE);

  std::cout << "out[1] = " << out[1] << ", out[999] = " << out[999] << std::endl;

  JSONLog::write("out", out);
  JSONLog::saveToFile("zout_"+backendName+".json");

  pImpl = nullptr;
  #ifdef USE_VULKAN
  vk_utils::globalContextDestroy();
  #endif
  return 0;
}
