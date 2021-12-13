#include "vk_swapchain.h"
#include "vk_utils.h"

std::vector<VkFramebuffer> vk_utils::createFrameBuffers(VkDevice a_device, const VulkanSwapChain &a_swapchain,
  VkRenderPass a_renderPass, VkImageView a_depthView)
{
  std::vector<VkFramebuffer> result(a_swapchain.GetImageCount());
  for (uint32_t i = 0; i < result.size(); i++)
  {
    std::vector<VkImageView> attachments;
    attachments.push_back(a_swapchain.GetAttachment(i).view);
    if(a_depthView != VK_NULL_HANDLE)
    {
      attachments.push_back(a_depthView);
    }

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType                   = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass              = a_renderPass;
    framebufferInfo.attachmentCount         = (uint32_t)attachments.size();
    framebufferInfo.pAttachments            = attachments.data();
    framebufferInfo.width                   = a_swapchain.GetExtent().width;
    framebufferInfo.height                  = a_swapchain.GetExtent().height;
    framebufferInfo.layers                  = 1;

    VK_CHECK_RESULT(vkCreateFramebuffer(a_device, &framebufferInfo, nullptr, &result[i]));
  }
  return result;
}


SwapChainSupportDetails vk_utils::querySwapChainSupport(VkPhysicalDevice device, const VkSurfaceKHR &surface)
{
  SwapChainSupportDetails details;

  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

  if (formatCount != 0)
  {
    details.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
  }

  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

  if (presentModeCount != 0)
  {
    details.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
  }

  return details;
}

VkQueue VulkanSwapChain::CreateSwapChain(const VkPhysicalDevice &physicalDevice, const VkDevice &logicalDevice, VkSurfaceKHR &surface,
                                         uint32_t &width, uint32_t &height, uint32_t imageCount, bool vsync)
{
  m_physicalDevice = physicalDevice;
  m_logicalDevice = logicalDevice;
  m_surface = surface;

  InitSurface(surface);
  Create(width, height, imageCount, vsync);

  VkQueue presentQueue;
  vkGetDeviceQueue(logicalDevice, m_queuePresentIndex, 0, &presentQueue);

  return presentQueue;
}

void VulkanSwapChain::InitSurface(VkSurfaceKHR &surface)
{
  uint32_t queueCount;
  vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueCount, nullptr);
  assert(queueCount >= 1);

  std::vector<VkQueueFamilyProperties> queueProps(queueCount);
  vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueCount, queueProps.data());

  std::vector<VkBool32> supportsPresent(queueCount);

  for (uint32_t i = 0; i < queueCount; ++i)
  {
    vkGetPhysicalDeviceSurfaceSupportKHR(m_physicalDevice, i, surface, &supportsPresent[i]);
  }

  // Search for a graphics and a present queue in the array of queue
  // families, try to find one that supports both
  uint32_t graphicsQueueNodeIndex = UINT32_MAX;
  uint32_t presentQueueNodeIndex  = UINT32_MAX;
  for (uint32_t i = 0; i < queueCount; i++)
  {
    if ((queueProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0)
    {
      if (graphicsQueueNodeIndex == UINT32_MAX)
      {
        graphicsQueueNodeIndex = i;
      }

      if (supportsPresent[i] == VK_TRUE)
      {
        graphicsQueueNodeIndex = i;
        presentQueueNodeIndex = i;
        break;
      }
    }
  }
  if (presentQueueNodeIndex == UINT32_MAX)
  {
    // If there's no queue that supports both present and graphics
    // try to find a separate present queue
    for (uint32_t i = 0; i < queueCount; ++i)
    {
      if (supportsPresent[i] == VK_TRUE)
      {
        presentQueueNodeIndex = i;
        break;
      }
    }
  }

  if (graphicsQueueNodeIndex == UINT32_MAX || presentQueueNodeIndex == UINT32_MAX)
  {
    RUN_TIME_ERROR("Could not find a graphics and/or presenting queue");
  }

  m_queueGraphicsIndex = graphicsQueueNodeIndex;
  m_queuePresentIndex = presentQueueNodeIndex;

  uint32_t formatCount;
  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, surface, &formatCount, nullptr);

  std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
  assert(formatCount > 0);
  vkGetPhysicalDeviceSurfaceFormatsKHR(m_physicalDevice, surface, &formatCount, surfaceFormats.data());

  auto surfFormat = ChooseSurfaceFormat(surfaceFormats);
  m_colorFormat = surfFormat.format;
  m_colorSpace = surfFormat.colorSpace;

}
VkSurfaceFormatKHR VulkanSwapChain::ChooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
  if (availableFormats.size() == 1 && availableFormats[0].format == VK_FORMAT_UNDEFINED)
  {
    return { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
  }

  for (const auto& availableFormat : availableFormats)
  {
    if (availableFormat.format == VK_FORMAT_B8G8R8A8_UNORM && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
    {
      return availableFormat;
    }
  }

  return availableFormats[0];
}

VkPresentModeKHR VulkanSwapChain::ChoosePresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes, bool vsync)
{
  VkPresentModeKHR bestMode = VK_PRESENT_MODE_FIFO_KHR;

  if (!vsync)
  {
    for (const auto &availablePresentMode : availablePresentModes)
    {
      if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR)
      {
        return availablePresentMode;
      }
      else if (availablePresentMode == VK_PRESENT_MODE_IMMEDIATE_KHR)
      {
        bestMode = availablePresentMode;
      }
    }
  }

  return bestMode;
}


/**
* Create the swapchain and get it's images with given width and height
*
* @param width Pointer to the width of the swapchain (may be adjusted to fit the requirements of the swapchain)
* @param height Pointer to the height of the swapchain (may be adjusted to fit the requirements of the swapchain)
* @param vsync (Optional) Can be used to force vsync'd rendering (by using VK_PRESENT_MODE_FIFO_KHR as presentation mode)
*/
void VulkanSwapChain::Create(uint32_t &width, uint32_t &height, uint32_t imageCount, bool vsync)
{
  VkSwapchainKHR oldSwapchain = m_swapChain;

  // Get physical device surface properties and formats
  VkSurfaceCapabilitiesKHR surfaceCaps;
  vkGetPhysicalDeviceSurfaceCapabilitiesKHR(m_physicalDevice, m_surface, &surfaceCaps);

  // Get available present modes
  uint32_t presentModeCount;
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, nullptr);
  assert(presentModeCount > 0);
  std::vector<VkPresentModeKHR> presentModes(presentModeCount);
  vkGetPhysicalDeviceSurfacePresentModesKHR(m_physicalDevice, m_surface, &presentModeCount, presentModes.data());

  // If width (and height) equals the special value 0xFFFFFFFF, the size of the surface will be set by the swapchain
  if (surfaceCaps.currentExtent.width == (uint32_t)-1)
  {
    m_swapchainExtent.width = width;
    m_swapchainExtent.height = height;
  }
  else
  {
    m_swapchainExtent = surfaceCaps.currentExtent;
    width = surfaceCaps.currentExtent.width;
    height = surfaceCaps.currentExtent.height;
  }

  VkPresentModeKHR swapchainPresentMode = ChoosePresentMode(presentModes, vsync);

  m_minImageCount = surfaceCaps.minImageCount;
  uint32_t desiredNumberOfSwapchainImages = std::max(imageCount, surfaceCaps.minImageCount);
  if ((surfaceCaps.maxImageCount > 0) && (desiredNumberOfSwapchainImages > surfaceCaps.maxImageCount))
  {
    desiredNumberOfSwapchainImages = surfaceCaps.maxImageCount;
  }

  // Find the transformation of the surface
  VkSurfaceTransformFlagsKHR preTransform = surfaceCaps.currentTransform;

#ifdef __ANDROID__
  if (surfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR ||
  surfaceCaps.currentTransform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR)
  {
    // Swap to get identity width and height
    surfaceCaps.currentExtent.height = width;
    surfaceCaps.currentExtent.width = height;
  }
#else
  if (surfaceCaps.supportedTransforms & VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR)
  {
    preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  }
  else
  {
    preTransform = surfaceCaps.currentTransform;
  }
#endif

  if (preTransform & VK_SURFACE_TRANSFORM_ROTATE_90_BIT_KHR)
  {
    m_surfaceMatrix = CreateSurfaceRotationMatrixZ(90.0f * DEG_TO_RAD);
  }
  else if (preTransform & VK_SURFACE_TRANSFORM_ROTATE_270_BIT_KHR)
  {
    m_surfaceMatrix = CreateSurfaceRotationMatrixZ(270.0f * DEG_TO_RAD);
  }
  else if (preTransform & VK_SURFACE_TRANSFORM_ROTATE_180_BIT_KHR)
  {
    m_surfaceMatrix = CreateSurfaceRotationMatrixZ(180.0f * DEG_TO_RAD);
  }

  VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

  // Select the first available composite alpha format
  std::vector<VkCompositeAlphaFlagBitsKHR> compositeAlphaFlags = {
          VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
          VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
          VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
          VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
  };
  for (auto& compositeAlphaFlag : compositeAlphaFlags)
  {
    if (surfaceCaps.supportedCompositeAlpha & compositeAlphaFlag)
    {
      compositeAlpha = compositeAlphaFlag;
      break;
    };
  }

  VkSwapchainCreateInfoKHR swapchainCreateInfo = {};
  swapchainCreateInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
  swapchainCreateInfo.pNext = nullptr;
  swapchainCreateInfo.surface = m_surface;
  swapchainCreateInfo.minImageCount = desiredNumberOfSwapchainImages;
  swapchainCreateInfo.imageFormat = m_colorFormat;
  swapchainCreateInfo.imageColorSpace = m_colorSpace;
  swapchainCreateInfo.imageExtent = { m_swapchainExtent.width, m_swapchainExtent.height };
  swapchainCreateInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  swapchainCreateInfo.preTransform = (VkSurfaceTransformFlagBitsKHR)preTransform;
  swapchainCreateInfo.imageArrayLayers = 1;

  uint32_t queueFamilyIndices[] = { m_queueGraphicsIndex, m_queuePresentIndex };

  if (m_queueGraphicsIndex != m_queuePresentIndex)
  {
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    swapchainCreateInfo.queueFamilyIndexCount = 2;
    swapchainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
  }
  else
  {
    swapchainCreateInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  }

  swapchainCreateInfo.presentMode = swapchainPresentMode;
  swapchainCreateInfo.oldSwapchain = oldSwapchain;
  // Setting clipped to VK_TRUE allows the implementation to discard rendering outside of the surface area
  swapchainCreateInfo.clipped = VK_TRUE;
  swapchainCreateInfo.compositeAlpha = compositeAlpha;

  if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
  {
    swapchainCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  }

  if (surfaceCaps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT)
  {
    swapchainCreateInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
  }

  vkCreateSwapchainKHR(m_logicalDevice, &swapchainCreateInfo, nullptr, &m_swapChain);

  // If an existing swap chain is re-created, destroy the old swap chain
  // This also cleans up all the presentable images
  if (oldSwapchain != VK_NULL_HANDLE)
  {
    for (uint32_t i = 0; i < m_imageCount; i++)
    {
      vkDestroyImageView(m_logicalDevice, m_attachments[i].view, nullptr);
    }
    vkDestroySwapchainKHR(m_logicalDevice, oldSwapchain, nullptr);
  }
  vkGetSwapchainImagesKHR(m_logicalDevice, m_swapChain, &m_imageCount, NULL);

  std::vector<VkImage> tmp_image_vec(m_imageCount);
  vkGetSwapchainImagesKHR(m_logicalDevice, m_swapChain, &m_imageCount, tmp_image_vec.data());

  m_attachments.resize(m_imageCount);
  for (uint32_t i = 0; i < m_imageCount; i++)
  {
    VkImageViewCreateInfo colorAttachmentView = {};
    colorAttachmentView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    colorAttachmentView.pNext = nullptr;
    colorAttachmentView.format = m_colorFormat;
    colorAttachmentView.components = {
            VK_COMPONENT_SWIZZLE_R,
            VK_COMPONENT_SWIZZLE_G,
            VK_COMPONENT_SWIZZLE_B,
            VK_COMPONENT_SWIZZLE_A
    };
    colorAttachmentView.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    colorAttachmentView.subresourceRange.baseMipLevel = 0;
    colorAttachmentView.subresourceRange.levelCount = 1;
    colorAttachmentView.subresourceRange.baseArrayLayer = 0;
    colorAttachmentView.subresourceRange.layerCount = 1;
    colorAttachmentView.viewType = VK_IMAGE_VIEW_TYPE_2D;
    colorAttachmentView.flags = 0;

    m_attachments[i].image = tmp_image_vec[i];
    m_attachments[i].format = m_colorFormat;

    colorAttachmentView.image = m_attachments[i].image;

    vkCreateImageView(m_logicalDevice, &colorAttachmentView, nullptr, &m_attachments[i].view);
  }
}

/**
  * Acquires the next image in the swap chain
  *
  * @param presentCompleteSemaphore (Optional) Semaphore that is signaled when the image is ready for use
  * @param imageIndex Pointer to the image index that will be increased if the next image could be acquired
  *
  * @note The function will always wait until the next image has been acquired by setting timeout to UINT64_MAX
  *
  * @return VkResult of the image acquisition
  */
VkResult VulkanSwapChain::AcquireNextImage(VkSemaphore semaphore, uint32_t *imageIndex)
{
  return vkAcquireNextImageKHR(m_logicalDevice, m_swapChain, UINT64_MAX, semaphore, VK_NULL_HANDLE, imageIndex);
}

/**
* Queue an image for presentation
*
* @param queue Presentation queue for presenting the image
* @param imageIndex Index of the swapchain image to queue for presentation
* @param waitSemaphore (Optional) Semaphore that is waited on before the image is presented (only used if != VK_NULL_HANDLE)
*
* @return VkResult of the queue presentation
*/
VkResult VulkanSwapChain::QueuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore)
{
  VkPresentInfoKHR presentInfo = {};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.pNext = NULL;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &m_swapChain;
  presentInfo.pImageIndices = &imageIndex;

  if (waitSemaphore != VK_NULL_HANDLE)
  {
    presentInfo.pWaitSemaphores = &waitSemaphore;
    presentInfo.waitSemaphoreCount = 1;
  }
  return vkQueuePresentKHR(queue, &presentInfo);
}

void VulkanSwapChain::Cleanup()
{
  if (m_swapChain != VK_NULL_HANDLE)
  {
    for (uint32_t i = 0; i < m_imageCount; i++)
    {
      vkDestroyImageView(m_logicalDevice, m_attachments[i].view, nullptr);
      m_attachments[i].view = VK_NULL_HANDLE;
    }
    vkDestroySwapchainKHR(m_logicalDevice, m_swapChain, nullptr);
    m_swapChain = VK_NULL_HANDLE;
  }
}

