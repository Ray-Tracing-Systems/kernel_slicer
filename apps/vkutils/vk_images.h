#ifndef VK_SORT_VK_TEXTURES_H
#define VK_SORT_VK_TEXTURES_H

#define USE_VOLK
#include "vk_include.h"
#include "vk_copy.h"

#include <string>
#include <vector>
#include <cassert>
#include <memory>
#include <cmath>

namespace vk_utils
{
  struct VulkanImageMem
  {
    VkFormat format;
    VkImageAspectFlags aspectMask;
    VkImage image;
    VkImageView view;

    VkDeviceMemory mem;
    VkDeviceSize mem_offset;
    VkMemoryRequirements memReq;
  };

  VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &depthFormats, VkFormat *depthFormat);
  bool isDepthFormat(VkFormat a_format);
  bool isStencilFormat(VkFormat a_format);
  bool isDepthOrStencil(VkFormat a_format);

  VulkanImageMem createImg(VkDevice a_device, uint32_t a_width, uint32_t a_height, VkFormat a_format, VkImageUsageFlags a_usage);
  VulkanImageMem createDepthTexture(VkDevice a_device, VkPhysicalDevice a_physDevice,
    const uint32_t a_width, const uint32_t a_height, VkFormat a_format);
  void deleteImg(VkDevice a_device, VulkanImageMem *a_pImgMem);


  VkImageView createImageViewAndBindMem(VkDevice a_device, VulkanImageMem *a_pImgMem, const VkImageViewCreateInfo *a_pViewCreateInfo = nullptr);

  void createImgAllocAndBind(VkDevice a_device, VkPhysicalDevice a_physicalDevice,
                             uint32_t a_width, uint32_t a_height, VkFormat a_format,  VkImageUsageFlags a_usage,
                             VulkanImageMem *a_pImgMem,
                             const VkImageCreateInfo *a_pImageCreateInfo = nullptr, const VkImageViewCreateInfo *a_pViewCreateInfo = nullptr);

  VulkanImageMem allocateColorTextureFromDataLDR(VkDevice a_device, VkPhysicalDevice a_physDevice, const unsigned char *pixels,
                                                 uint32_t w, uint32_t h, uint32_t a_mipLevels, VkFormat a_format,
                                                 std::shared_ptr<vk_utils::ICopyEngine> a_pCopy,
                                                 VkImageUsageFlags a_usageFlags = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

  VkSampler createSampler(VkDevice a_device,
                          VkFilter a_filterMode = VK_FILTER_LINEAR,
                          VkSamplerAddressMode a_addressMode = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                          VkBorderColor a_border_color = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE,
                          uint32_t a_mipLevels = 1);

  inline uint32_t calcMipLevelsCount(uint32_t w, uint32_t h)
  {
    return static_cast<uint32_t>(floor(log2(std::max(w, h))) + 1);
  }

  void recordMipChainGenerationCmdBuf(VkDevice a_device, VkCommandBuffer a_cmdBuf, const VulkanImageMem& imageMem,
                                      uint32_t a_width, uint32_t a_height, uint32_t a_mipLevels, VkImageLayout a_targetLayout);

  // *** layout transitions and image barriers ***
  // taken from https://github.com/SaschaWillems/Vulkan
  void setImageLayout(
      VkCommandBuffer cmdBuffer,
      VkImage image,
      VkImageLayout oldImageLayout,
      VkImageLayout newImageLayout,
      VkImageSubresourceRange subresourceRange,
      VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  void setImageLayout(
      VkCommandBuffer cmdBuffer,
      VkImage image,
      VkImageAspectFlags aspectMask,
      VkImageLayout oldImageLayout,
      VkImageLayout newImageLayout,
      VkPipelineStageFlags srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
      VkPipelineStageFlags dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  void insertImageMemoryBarrier(
      VkCommandBuffer cmdBuffer,
      VkImage image,
      VkAccessFlags srcAccessMask,
      VkAccessFlags dstAccessMask,
      VkImageLayout oldImageLayout,
      VkImageLayout newImageLayout,
      VkPipelineStageFlags srcStageMask,
      VkPipelineStageFlags dstStageMask,
      VkImageSubresourceRange subresourceRange);
  // ****************
}


#endif //VK_SORT_VK_TEXTURES_H
