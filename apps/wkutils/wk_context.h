#pragma once

#ifdef USE_DAWN
#include <webgpu/webgpu.hpp>
#else
#include <webgpu/webgpu.h>
#include <webgpu/wgpu.h>
#endif

#define WGPU_DISTR 30

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
  void          printDeviceInfo(WGPUAdapter adapter);
  
  void          readBufferBack(WulkanContext a_ctx, WGPUQueue a_queue, WGPUBuffer a_buffer, WGPUBuffer a_tmpBuffer, size_t a_size, void* a_data);
};
