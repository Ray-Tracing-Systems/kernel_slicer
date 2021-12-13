#ifndef VK_SWAPCHAIN_H
#define VK_SWAPCHAIN_H

#include "vk_include.h"

#include <vector>
#include <array>
#include <cmath>
#include <cassert>

struct SwapchainAttachment
{
  VkFormat format;
  VkImage image = VK_NULL_HANDLE;
  VkImageView view = VK_NULL_HANDLE;
};


struct SwapChainSupportDetails
{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};


class VulkanSwapChain
{
public:

  VkQueue CreateSwapChain(const VkPhysicalDevice &physicalDevice, const VkDevice &logicalDevice, VkSurfaceKHR &a_surface,
                          uint32_t &width, uint32_t &height, uint32_t a_imageCount = 2, bool vsync = false);

  static VkSurfaceFormatKHR ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats);

  static VkPresentModeKHR ChoosePresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes, bool vsync);
    
  VkResult AcquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t *imageIndex);

  VkResult QueuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore = VK_NULL_HANDLE);

  void Cleanup();

  VkFormat GetFormat() const {return m_colorFormat; }
  uint32_t GetImageCount() const {return m_imageCount; }
  uint32_t GetMinImageCount() const {return m_minImageCount; }
  VkExtent2D GetExtent() const {return m_swapchainExtent; }
  SwapchainAttachment GetAttachment(uint32_t i) const { assert(i < m_imageCount); return m_attachments[i]; }

  std::array<float, 16> GetSurfaceMatrixArr() const {return m_surfaceMatrix; }
  const float* GetSurfaceMatrixPtr() const {return m_surfaceMatrix.data(); }
  VkSurfaceTransformFlagsKHR GetSurfaceTransformFlags() const { return m_surfaceTransformFlags; }

private:
  VkSwapchainKHR m_swapChain = VK_NULL_HANDLE;

  VkSurfaceTransformFlagsKHR m_surfaceTransformFlags = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  VkFormat m_colorFormat;
  uint32_t m_imageCount;
  uint32_t m_minImageCount;
  VkExtent2D m_swapchainExtent {};
  VkColorSpaceKHR m_colorSpace;

  std::vector<SwapchainAttachment> m_attachments;

  uint32_t m_queueGraphicsIndex = UINT32_MAX;
  uint32_t m_queuePresentIndex  = UINT32_MAX;

  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VkDevice m_logicalDevice = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;

  void Create(uint32_t &width, uint32_t &height, uint32_t imageCount, bool vsync = false);
  void InitSurface(VkSurfaceKHR &a_surface);

  // 4x4 matrix stored by columns
  std::array<float, 16> m_surfaceMatrix = {1.0f, 0.0f, 0.0, 0.0f,
                                           0.0f, 1.0f, 0.0, 0.0f,
                                           0.0f, 0.0f, 1.0, 0.0f,
                                           0.0f, 0.0f, 0.0, 1.0f};
  static constexpr float DEG_TO_RAD = 3.14159265358979323846f / 180.0f;
  static std::array<float, 16> CreateSurfaceRotationMatrixZ(float phi)
  {
    std::array<float, 16> res = {
      +cosf(phi), sinf(phi), 0.0f, 0.0f,
      -sinf(phi), cosf(phi), 0.0f, 0.0f,
      0.0f,            0.0f, 1.0f, 0.0f,
      0.0f,            0.0f, 0.0f, 1.0f
    };
    return res;
  }

};

namespace vk_utils
{
  std::vector<VkFramebuffer> createFrameBuffers(VkDevice a_device, const VulkanSwapChain &a_swapchain,
                                                VkRenderPass a_renderPass, VkImageView a_depthView = VK_NULL_HANDLE);

  SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device, const VkSurfaceKHR &surface);
}

#endif //VK_SWAPCHAIN_H
