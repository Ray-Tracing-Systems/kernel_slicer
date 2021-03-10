#pragma once

#if defined(__ANDROID__) // Dynamic load, use vulkan_wrapper.h to load vulkan functions
  #include "vulkan_wrapper/vulkan_wrapper.h"
#else
  #include <vulkan/vulkan.h>
#endif

namespace vkfw
{
  VkBuffer             CreateBuffer (VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags);
  VkDeviceMemory       AllocateAndBindWithPadding(VkDevice a_dev, VkPhysicalDevice a_physDev, const std::vector<VkBuffer> a_buffers);

  VkMemoryRequirements CreateBuffer (VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, VkBuffer& a_buf);
  std::vector<size_t>  AssignMemOffsetsWithPadding(const std::vector<VkMemoryRequirements> a_memInfos);
};


namespace vkfw
{
  struct BufferReqPair
  {
    VkBuffer             buffer = VK_NULL_HANDLE;
    VkMemoryRequirements req;
  };

  BufferReqPair        CreateBuffer2(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, VkBuffer& a_buf, const VkAllocationCallbacks* pAllocator = nullptr);
  BufferReqPair        CreateBuffer2(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, const VkAllocationCallbacks* pAllocator = nullptr);

  std::vector<size_t> AssignMemOffsetsWithPadding(const std::vector<BufferReqPair> a_memInfos);
};

