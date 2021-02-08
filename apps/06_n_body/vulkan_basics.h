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

static constexpr uint32_t KGEN_FLAG_RETURN = 1;
static constexpr uint32_t KGEN_FLAG_BREAK  = 2;
static constexpr uint32_t KGEN_FLAG_DONT_SET_EXIT     = 4;
static constexpr uint32_t KGEN_FLAG_SET_EXIT_NEGATIVE = 8;
static constexpr uint32_t KGEN_REDUCTION_LAST_STEP    = 16;

#endif
