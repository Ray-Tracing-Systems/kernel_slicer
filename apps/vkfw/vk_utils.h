//
// Created by frol on 15.06.19.
//

#ifndef VULKAN_MINIMAL_COMPUTE_VK_UTILS_H
#define VULKAN_MINIMAL_COMPUTE_VK_UTILS_H

#include <vulkan/vulkan.h>
#include "vk_swapchain.h"
#include <vector>

#include <stdexcept>
#include <sstream>
#include <cassert>
//#include <LiteMath.h>
//#include <aligned_alloc.h>

#if defined(__ANDROID__)
#include <android/log.h>
#endif

namespace vk_utils
{
  constexpr uint64_t FENCE_TIMEOUT = 100000000000l;
  typedef VkBool32 (VKAPI_PTR *DebugReportCallbackFuncType)(VkDebugReportFlagsEXT      flags,
                                                            VkDebugReportObjectTypeEXT objectType,
                                                            uint64_t                   object,
                                                            size_t                     location,
                                                            int32_t                    messageCode,
                                                            const char*                pLayerPrefix,
                                                            const char*                pMessage,
                                                            void*                      pUserData);


  static void RunTimeError(const char* file, int line, const char* msg)
  {
    std::stringstream strout;
    strout << "runtime_error at " << file << ", line " << line << ": " << msg << std::endl;
    throw std::runtime_error(strout.str().c_str());
  }

  struct queueFamilyIndices
  {
      uint32_t graphics;
      uint32_t compute;
      uint32_t transfer;
  };

  VkInstance CreateInstance(bool &a_enableValidationLayers, std::vector<const char *> &a_requestedLayers,
                            std::vector<const char *> &a_instanceExtensions, VkApplicationInfo* appInfo = nullptr);

  void       InitDebugReportCallback(VkInstance a_instance, DebugReportCallbackFuncType a_callback, VkDebugReportCallbackEXT* a_debugReportCallback);
  VkPhysicalDevice FindPhysicalDevice(VkInstance a_instance, bool a_printInfo, int a_preferredDeviceId, std::vector<const char *> a_deviceExt = {});

  uint32_t GetQueueFamilyIndex(VkPhysicalDevice a_physicalDevice, int a_bits);
  uint32_t GetComputeQueueFamilyIndex(VkPhysicalDevice a_physicalDevice);
  VkDevice CreateLogicalDevice(VkPhysicalDevice physicalDevice, const std::vector<const char *>& a_enabledLayers,
                               std::vector<const char *> a_extensions, VkPhysicalDeviceFeatures a_deviceFeatures,
                               queueFamilyIndices &a_queueIDXs, VkQueueFlags requestedQueueTypes = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT,
                               VkPhysicalDeviceFeatures2 a_deviceFeatures2 = {});
  uint32_t FindMemoryType(uint32_t memoryTypeBits, VkMemoryPropertyFlags properties, VkPhysicalDevice physicalDevice);

  size_t   Padding(size_t a_size, size_t a_alignment);

  std::string GetResourcesDir();
  std::string GetShadersDir(const std::string &sample_name);

  //// FrameBuffer and SwapChain issues
  //
  struct ScreenBufferResources
  {
    VkSurfaceKHR               surface;
    std::vector<VkFramebuffer> framebuffers;
    VulkanSwapChain swapchain;

    // screen depthbuffer
    //
    VkImage        depthImage       = VK_NULL_HANDLE;
    VkDeviceMemory depthImageMemory = VK_NULL_HANDLE;
    VkImageView    depthImageView   = VK_NULL_HANDLE;
    VkFormat       depthFormat;
    VkSurfaceTransformFlagsKHR preTransform; // android crap
  };

  void CreateSwapChain(VkPhysicalDevice a_physDevice, VkDevice a_device, int a_width, int a_height, ScreenBufferResources* pScreen, bool vsync = false);

  void CreateScreenImageViews(VkDevice a_device, ScreenBufferResources* pScreen);

  void CreateScreenFrameBuffers(VkDevice a_device, VkRenderPass a_renderPass, VkImageView a_depthView, ScreenBufferResources* pScreen);

  void CreateDepthTexture(VkDevice a_device, VkPhysicalDevice a_physDevice, const int a_width, const int a_height, VkFormat a_format,
                          VkDeviceMemory *a_pImageMemory, VkImage *a_image, VkImageView* a_imageView);
  VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &depthFormats, VkFormat *depthFormat);

  bool CreateRenderPass(VkDevice a_device, VkFormat a_swapChainImageFormat, VkRenderPass* a_pRenderPass);

#if defined(__ANDROID__)
  std::vector<uint32_t> ReadFile(AAssetManager* mgr, const char* filename);
#endif
  std::vector<uint32_t> ReadFile(const char* filename);
  VkShaderModule CreateShaderModule(VkDevice a_device, const std::vector<uint32_t>& code);

  /**
  \brief Immediately execute command buffer and wait.
  */
  void ExecuteCommandBufferNow(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device);

  /**
  \brief TBD
  */
  VkCommandPool                CreateCommandPool(VkDevice a_device, VkPhysicalDevice a_physDevice, VkQueueFlagBits a_queueFlags, VkCommandPoolCreateFlagBits a_poolFlags);
  
  /**
  \brief TBD
  */
  std::vector<VkCommandBuffer> CreateCommandBuffers(VkDevice a_device, VkCommandPool a_pool, uint32_t a_buffNum);

  bool IsDepthFormat(VkFormat a_format);

  struct IMemBuf : std::streambuf // http://www.cplusplus.com/forum/general/226786/
  {
      IMemBuf(const char* base, size_t size)
      {
        char* p(const_cast<char*>(base));
        this->setg(p, p, p + size);
      }
  };

  struct IMemStream : virtual IMemBuf, std::istream
  {
      IMemStream(const char* mem, size_t size) : IMemBuf(mem, size), std::istream(static_cast<std::streambuf*>(this)) {}
  };
};

#undef  RUN_TIME_ERROR
#undef  RUN_TIME_ERROR_AT
#define RUN_TIME_ERROR(e) (vk_utils::RunTimeError(__FILE__,__LINE__,(e)))
#define RUN_TIME_ERROR_AT(e, file, line) (vk_utils::RunTimeError((file),(line),(e)))

// Used for validating return values of Vulkan API calls.
//
#if defined(__ANDROID__)
#define VK_CHECK_RESULT(f)                                                 \
  if (VK_SUCCESS != (f)) {                                         \
    __android_log_print(ANDROID_LOG_ERROR, "VulkanAPP ",               \
                        "Vulkan error. File[%s], line[%d]", __FILE__, \
                        __LINE__);                                    \
    assert(false);                                                    \
  }
#else
#define VK_CHECK_RESULT(f) 													\
{																										\
    VkResult _res = (f);														\
    if (_res != VK_SUCCESS)													\
    {																								\
        printf("Fatal : VkResult is %d in %s at line %d\n", _res,  __FILE__, __LINE__); \
        assert(_res == VK_SUCCESS);									\
    }																								\
}
#endif

#endif //VULKAN_MINIMAL_COMPUTE_VK_UTILS_H
