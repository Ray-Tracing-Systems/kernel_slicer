#pragma once

#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>

namespace wk_utils
{
  struct WulkanDeviceFeatures
  {
    int dummy;
  };

  struct WulkanContext
  {
    WGPUInstance instance       = nullptr;
    WGPUAdapter  physicalDevice = nullptr;
    WGPUDevice   device         = nullptr;
  };

  WulkanContext globalContextInit(WulkanDeviceFeatures a_features);

  void printDeviceInfo(WGPUAdapter adapter);
};
