#include "vk_context.h"
#include "vk_utils.h"
#include "vk_buffers.h"

#include <cstring>
#include <cassert>
#include <iostream>

#include <cmath>
#include <cassert>

#include <algorithm>
#ifdef WIN32
#undef min
#undef max
#endif 

vk_utils::VulkanContext g_ctx;

bool vk_utils::globalContextIsInitialized(const std::vector<const char*>& requiredExtensions)
{
  // todo: if initialized check that requiredExtensions were actually initialized
  // if context is initialized but device featurea are wrong, print error and exit
  return (g_ctx.instance != VK_NULL_HANDLE) && (g_ctx.physicalDevice != VK_NULL_HANDLE) && (g_ctx.device != VK_NULL_HANDLE);
}

vk_utils::VulkanContext vk_utils::globalContextGet(bool enableValidationLayers, unsigned int a_preferredDeviceId)
{
  if(globalContextIsInitialized())
    return g_ctx;
  else
    return globalContextInit(std::vector<const char*>(), enableValidationLayers, a_preferredDeviceId);
}

struct RTXDeviceFeatures
{
  VkPhysicalDeviceAccelerationStructureFeaturesKHR m_enabledAccelStructFeatures{};
  VkPhysicalDeviceBufferDeviceAddressFeatures      m_enabledDeviceAddressFeatures{};
  VkPhysicalDeviceRayQueryFeaturesKHR              m_enabledRayQueryFeatures;
};

static RTXDeviceFeatures SetupRTXFeatures(VkPhysicalDevice a_physDev)
{
  static RTXDeviceFeatures g_rtFeatures;

  g_rtFeatures.m_enabledRayQueryFeatures.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
  g_rtFeatures.m_enabledRayQueryFeatures.rayQuery = VK_TRUE;

  g_rtFeatures.m_enabledDeviceAddressFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  g_rtFeatures.m_enabledDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
  g_rtFeatures.m_enabledDeviceAddressFeatures.pNext               = &g_rtFeatures.m_enabledRayQueryFeatures;

  g_rtFeatures.m_enabledAccelStructFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  g_rtFeatures.m_enabledAccelStructFeatures.accelerationStructure = VK_TRUE;
  g_rtFeatures.m_enabledAccelStructFeatures.pNext                 = &g_rtFeatures.m_enabledDeviceAddressFeatures;

  return g_rtFeatures;
}

vk_utils::VulkanContext vk_utils::globalContextInit(const std::vector<const char*>& requiredExtensions, bool enableValidationLayers, unsigned int a_preferredDeviceId)
{
  if(globalContextIsInitialized(requiredExtensions))
    return g_ctx;

  std::vector<const char*> enabledLayers;
  std::vector<const char*> extensions;
  enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
  enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
  VK_CHECK_RESULT(volkInitialize());
  g_ctx.instance = vk_utils::createInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(g_ctx.instance);

  g_ctx.physicalDevice = vk_utils::findPhysicalDevice(g_ctx.instance, true, a_preferredDeviceId);
  auto queueComputeFID = vk_utils::getQueueFamilyIndex(g_ctx.physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // query features for RTX
  //
  RTXDeviceFeatures rtxFeatures = SetupRTXFeatures(g_ctx.physicalDevice);

  // query features for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;
  features.pNext      = &rtxFeatures.m_enabledAccelStructFeatures;

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;
  
  vk_utils::QueueFID_T fIDs = {};
  
  if(requiredExtensions.size() == 0) // TODO: remove this or work around
  {
    deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
    deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 

    // Required by VK_KHR_RAY_QUERY
    deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    deviceExtensions.push_back("VK_KHR_spirv_1_4");
    deviceExtensions.push_back("VK_KHR_shader_float_controls");  
  
    // Required by VK_KHR_acceleration_structure
    deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
  
    // // Required by VK_KHR_ray_tracing_pipeline
    // m_deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
    // // Required by VK_KHR_spirv_1_4
    // m_deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  fIDs.compute = queueComputeFID;
  g_ctx.device = vk_utils::createLogicalDevice(g_ctx.physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures,
                                               fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT, &features);
  volkLoadDevice(g_ctx.device);                                            
  g_ctx.commandPool = vk_utils::createCommandPool(g_ctx.device, fIDs.compute, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // (2) initialize vulkan helpers
  //  
  {
    auto queueComputeFID = vk_utils::getQueueFamilyIndex(g_ctx.physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue(g_ctx.device, queueComputeFID, 0, &g_ctx.computeQueue);
    vkGetDeviceQueue(g_ctx.device, queueComputeFID, 0, &g_ctx.transferQueue);
  }

  g_ctx.pCopyHelper = std::make_shared<vk_utils::SimpleCopyHelper>(g_ctx.physicalDevice, g_ctx.device, g_ctx.transferQueue, queueComputeFID, 64*1024*1024); // TODO, select PinPong Helper by default!
  return g_ctx;
}

void vk_utils::globalContextDestroy()
{
  if(g_ctx.device == VK_NULL_HANDLE)
    return;
  vkDestroyCommandPool(g_ctx.device, g_ctx.commandPool, nullptr);
  vkDestroyDevice(g_ctx.device, nullptr);
  vkDestroyInstance(g_ctx.instance, nullptr);
  g_ctx.device         = VK_NULL_HANDLE;
  g_ctx.physicalDevice = VK_NULL_HANDLE;
  g_ctx.instance       = VK_NULL_HANDLE;
}
