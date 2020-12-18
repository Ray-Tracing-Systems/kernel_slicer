#ifndef VULKAN_BASICS_H
#define VULKAN_BASICS_H

#include "vk_copy.h"

struct VulkanContext
{
  VkInstance       instance       = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;
  VkCommandPool    commandPool    = VK_NULL_HANDLE; 

  VkQueue computeQueue  = VK_NULL_HANDLE;
  VkQueue transferQueue = VK_NULL_HANDLE;
};

static constexpr uint32_t KGEN_OUTSIDE_OF_FOR = 1;
static constexpr uint32_t KGEN_INSIDE_FOR     = 2;

#endif