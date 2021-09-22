#ifndef VK_UTILS_H
#define VK_UTILS_H

#if defined(__ANDROID__)
#include <android_native_app_glue.h>
#include <android/log.h>

#endif

#define USE_VOLK
#include "vk_include.h"

#include <string>
#include <iostream>
#include <vector>
#include <cassert>
#include <sstream>

namespace vk_utils
{
  constexpr uint64_t DEFAULT_TIMEOUT = 100000000000l;
  constexpr VkColorComponentFlags ALL_COLOR_COMPONENTS = VK_COLOR_COMPONENT_R_BIT |
                                                         VK_COLOR_COMPONENT_G_BIT |
                                                         VK_COLOR_COMPONENT_B_BIT |
                                                         VK_COLOR_COMPONENT_A_BIT; // 0xF

  struct QueueFID_T
  {
    uint32_t graphics;
    uint32_t compute;
    uint32_t transfer;
  };

  VkInstance createInstance(bool &a_enableValidationLayers, std::vector<const char *> &a_requestedLayers,
                            std::vector<const char *> &a_instanceExtensions, VkApplicationInfo* appInfo = nullptr);

  VkPhysicalDevice findPhysicalDevice(VkInstance a_instance, bool a_printInfo, unsigned a_preferredDeviceId, std::vector<const char *> a_deviceExt = {});

  VkDevice createLogicalDevice(VkPhysicalDevice physicalDevice, const std::vector<const char *>& a_enabledLayers,
                               std::vector<const char *> a_extensions, VkPhysicalDeviceFeatures a_deviceFeatures,
                               QueueFID_T &a_queueIDXs, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT,
                               void* pNextFeatures = nullptr);
  uint32_t findMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice);
  uint32_t getQueueFamilyIndex(VkPhysicalDevice a_physicalDevice, VkQueueFlags a_bits);
  std::vector<std::string> subgroupOperationToString(VkSubgroupFeatureFlags flags);

  size_t getPaddedSize(size_t a_size, size_t a_alignment);
  uint32_t getSBTAlignedSize(uint32_t value, uint32_t alignment);

#ifdef __ANDROID__
  std::vector<uint32_t> readSPVFile(AAssetManager* mgr, const char* filename);
#else
  std::vector<uint32_t> readSPVFile(const char* filename);
#endif
  VkShaderModule createShaderModule(VkDevice a_device, const std::vector<uint32_t>& code);
  VkPipelineShaderStageCreateInfo loadShader(VkDevice a_device, const std::string& fileName, VkShaderStageFlagBits stage,
                                             std::vector<VkShaderModule> &modules);

  // *** commands ***
  //
  VkCommandPool createCommandPool(VkDevice a_device,  uint32_t a_queueIdx, VkCommandPoolCreateFlagBits a_poolFlags);
  VkCommandBuffer createCommandBuffer(VkDevice a_device, VkCommandPool a_pool);
  std::vector<VkCommandBuffer> createCommandBuffers(VkDevice a_device, VkCommandPool a_pool, uint32_t a_buffNum);

  void executeCommandBufferNow(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device);
  void executeCommandBufferNow(std::vector<VkCommandBuffer> a_cmdBuffers, VkQueue a_queue, VkDevice a_device);
  // ****************

  void setDefaultViewport(VkCommandBuffer a_cmdBuff, float a_width, float a_height);
  void setDefaultScissor(VkCommandBuffer a_cmdBuff, uint32_t a_width, uint32_t a_height);

  // *** render pass ***
  //
  struct RenderTargetInfo2D
  {
    VkExtent2D         size;
    VkFormat           format;
    VkAttachmentLoadOp loadOp;
    VkImageLayout      initialLayout;
    VkImageLayout      finalLayout;
  };

  VkRenderPass createDefaultRenderPass(VkDevice a_device, VkFormat a_imageFormat);
  VkRenderPass createRenderPass(VkDevice a_device, RenderTargetInfo2D a_rtInfo);
  // ****************

  // *** errors and debugging ***
  //
  static FILE* log = stderr;

  void setLogToFile(const std::string &path);
  void runTimeError(const char* file, int line, const char* msg);
  void logWarning(const std::string& msg);
  void logInfo(const std::string& msg);
  std::string errorString(VkResult errorCode);

  typedef VkBool32 (VKAPI_PTR *DebugReportCallbackFuncType)(VkDebugReportFlagsEXT      flags,
                                                            VkDebugReportObjectTypeEXT objectType,
                                                            uint64_t                   object,
                                                            size_t                     location,
                                                            int32_t                    messageCode,
                                                            const char*                pLayerPrefix,
                                                            const char*                pMessage,
                                                            void*                      pUserData);

  void initDebugReportCallback(VkInstance a_instance, DebugReportCallbackFuncType a_callback, VkDebugReportCallbackEXT* a_debugReportCallback);
  // ****************
}


#ifdef __ANDROID__
#define VK_CHECK_RESULT(f) 													           \
{																										           \
    VkResult __vk_check_result = (f);													 \
    if (__vk_check_result != VK_SUCCESS)											 \
    {																								           \
        __android_log_print(ANDROID_LOG_ERROR, "vk_utils", "Fatal : VkResult is %s in %s at line %d\n",    \
                vk_utils::errorString(__vk_check_result).c_str(),  __FILE__, __LINE__); \
        assert(__vk_check_result == VK_SUCCESS);							 \
    }																								           \
}


#else

#define VK_CHECK_RESULT(f) 													           \
{																										           \
    VkResult __vk_check_result = (f);													 \
    if (__vk_check_result != VK_SUCCESS)											 \
    {																								           \
        fprintf(vk_utils::log, "Fatal : VkResult is %s in %s at line %d\n",    \
                vk_utils::errorString(__vk_check_result).c_str(),  __FILE__, __LINE__); \
        assert(__vk_check_result == VK_SUCCESS);							 \
    }																								           \
}

#endif

#undef  RUN_TIME_ERROR
#undef  RUN_TIME_ERROR_AT
#define RUN_TIME_ERROR(e) (vk_utils::runTimeError(__FILE__,__LINE__,(e)))
#define RUN_TIME_ERROR_AT(e, file, line) (vk_utils::runTimeError((file),(line),(e)))

#endif //VK_UTILS_H
