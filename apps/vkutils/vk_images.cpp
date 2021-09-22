#include "vk_images.h"
#include "vk_utils.h"

#include <array>
#include <algorithm>


namespace vk_utils
{

  VulkanImageMem createImg(VkDevice a_device, uint32_t a_width, uint32_t a_height, VkFormat a_format, VkImageUsageFlags a_usage)
  {
    VulkanImageMem result = {};

    result.format = a_format;
    if (a_usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
      result.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    if (a_usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
      result.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = a_format;
    image.extent.width = a_width;
    image.extent.height = a_height;
    image.extent.depth = 1;
    image.mipLevels = 1;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = a_usage;
    
    VK_CHECK_RESULT(vkCreateImage(a_device, &image, nullptr, &result.image));
    vkGetImageMemoryRequirements(a_device, result.image, &result.memReq);
    
    return result;
  }

  void deleteImg(VkDevice a_device, VulkanImageMem *a_pImgMem)
  {
    if(a_pImgMem->view != VK_NULL_HANDLE)
    {
      vkDestroyImageView(a_device, a_pImgMem->view, nullptr);
      a_pImgMem->view = VK_NULL_HANDLE;
    }

    if(a_pImgMem->image != VK_NULL_HANDLE)
    {
      vkDestroyImage(a_device, a_pImgMem->image, nullptr);
      a_pImgMem->image = VK_NULL_HANDLE;
    }
  }

  VkImageView createImageViewAndBindMem(VkDevice a_device, VulkanImageMem *a_pImgMem, const VkImageViewCreateInfo *a_pViewCreateInfo)
  {
    VK_CHECK_RESULT(vkBindImageMemory(a_device, a_pImgMem->image, a_pImgMem->mem, a_pImgMem->mem_offset));

    VkImageViewCreateInfo imageView{};
    if(a_pViewCreateInfo != nullptr)
    {
      imageView = *a_pViewCreateInfo;
    }
    else
    {
      imageView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      imageView.viewType = VK_IMAGE_VIEW_TYPE_2D;
      imageView.format = a_pImgMem->format;
      imageView.subresourceRange = {};
      imageView.subresourceRange.aspectMask = a_pImgMem->aspectMask;
      imageView.subresourceRange.baseMipLevel = 0;
      imageView.subresourceRange.levelCount = 1;
      imageView.subresourceRange.baseArrayLayer = 0;
      imageView.subresourceRange.layerCount = 1;
    }
    imageView.image = a_pImgMem->image;

    VK_CHECK_RESULT(vkCreateImageView(a_device, &imageView, nullptr, &a_pImgMem->view));
    return a_pImgMem->view;
  }

  VulkanImageMem allocateColorTextureFromDataLDR(VkDevice a_device, VkPhysicalDevice a_physDevice, const unsigned char *pixels,
                                                 uint32_t w, uint32_t h, uint32_t a_mipLevels, VkFormat a_format,
                                                 std::shared_ptr<vk_utils::ICopyEngine> a_pCopy, VkImageUsageFlags a_usageFlags)
  {
    VulkanImageMem result = {};

    result.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    result.format     = a_format;

    VkImageCreateInfo image{};
    image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType = VK_IMAGE_TYPE_2D;
    image.format = result.format;
    image.extent.width = w;
    image.extent.height = h;
    image.extent.depth = 1;
    image.mipLevels = a_mipLevels;
    image.arrayLayers = 1;
    image.samples = VK_SAMPLE_COUNT_1_BIT;
    image.tiling = VK_IMAGE_TILING_OPTIMAL;
    image.usage = a_usageFlags | VK_IMAGE_USAGE_SAMPLED_BIT;

    VK_CHECK_RESULT(vkCreateImage(a_device, &image, nullptr, &result.image));
    vkGetImageMemoryRequirements(a_device, result.image, &result.memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize  = result.memReq.size;
    memAlloc.memoryTypeIndex = findMemoryType(result.memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice);
    VK_CHECK_RESULT(vkAllocateMemory(a_device, &memAlloc, nullptr, &result.mem));

    createImageViewAndBindMem(a_device, &result);

    a_pCopy->UpdateImage(result.image, pixels, w, h, 4);

    return result;
  }

  void createImgAllocAndBind(VkDevice a_device, VkPhysicalDevice a_physicalDevice,
                             uint32_t a_width, uint32_t a_height, VkFormat a_format,  VkImageUsageFlags a_usage,
                             VulkanImageMem *a_pImgMem,
                             const VkImageCreateInfo *a_pImageCreateInfo, const VkImageViewCreateInfo *a_pViewCreateInfo)
  {

    a_pImgMem->format = a_format;
    // aspect mask ?
    VkImageCreateInfo image{};
    if(a_pImageCreateInfo != nullptr)
    {
      image = *a_pImageCreateInfo;
    }
    else
    {
      image.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
      image.imageType = VK_IMAGE_TYPE_2D;
      image.format = a_format;
      image.extent.width = a_width;
      image.extent.height = a_height;
      image.extent.depth = 1;
      image.mipLevels = 1;
      image.arrayLayers = 1;
      image.samples = VK_SAMPLE_COUNT_1_BIT;
      image.tiling = VK_IMAGE_TILING_OPTIMAL;
      image.usage = a_usage | VK_IMAGE_USAGE_SAMPLED_BIT;
    }

    VK_CHECK_RESULT(vkCreateImage(a_device, &image, nullptr, &a_pImgMem->image));
    vkGetImageMemoryRequirements(a_device, a_pImgMem->image, &a_pImgMem->memReq);

    VkMemoryAllocateInfo memAlloc{};
    memAlloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memAlloc.allocationSize  = a_pImgMem->memReq.size;
    memAlloc.memoryTypeIndex = findMemoryType(a_pImgMem->memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(a_device, &memAlloc, nullptr, &a_pImgMem->mem));

    createImageViewAndBindMem(a_device, a_pImgMem, a_pViewCreateInfo);
  }

  VkBool32 getSupportedDepthFormat(VkPhysicalDevice physicalDevice, const std::vector<VkFormat> &depthFormats, VkFormat *depthFormat)
  {
    for (auto &format : depthFormats)
    {
      VkFormatProperties formatProps;
      vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &formatProps);

      // Format must support depth stencil attachment for optimal tiling
      if (formatProps.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)
      {
        *depthFormat = format;
        return true;
      }
    }

    return false;
  }

  bool isDepthFormat(VkFormat a_format)
  {
    static constexpr std::array<VkFormat, 6> depth_formats =
        {
            VK_FORMAT_D16_UNORM,
            VK_FORMAT_X8_D24_UNORM_PACK32,
            VK_FORMAT_D32_SFLOAT,
            VK_FORMAT_D16_UNORM_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
        };

    return std::find(depth_formats.begin(), depth_formats.end(), a_format) != std::end(depth_formats);
  }

  bool isStencilFormat(VkFormat a_format)
  {
    static constexpr std::array<VkFormat, 4> stencil_formats =
        {
            VK_FORMAT_S8_UINT,
            VK_FORMAT_D16_UNORM_S8_UINT,
            VK_FORMAT_D24_UNORM_S8_UINT,
            VK_FORMAT_D32_SFLOAT_S8_UINT,
        };
    return std::find(stencil_formats.begin(), stencil_formats.end(), a_format) != std::end(stencil_formats);
  }

  bool isDepthOrStencil(VkFormat a_format)
  {
    return(isDepthFormat(a_format) || isStencilFormat(a_format));
  }

  VulkanImageMem createDepthTexture(VkDevice a_device, VkPhysicalDevice a_physDevice,
    const uint32_t a_width, const uint32_t a_height, VkFormat a_format)
  {
    VulkanImageMem result = {};
    result.format = a_format;

    VkImageCreateInfo imgCreateInfo = {};
    imgCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCreateInfo.pNext = nullptr;
    imgCreateInfo.flags = 0;
    imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imgCreateInfo.format = a_format;
    imgCreateInfo.extent = VkExtent3D{ a_width, a_height, 1u };
    imgCreateInfo.mipLevels = 1;
    imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCreateInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    imgCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCreateInfo.arrayLayers = 1;

    VkImageViewCreateInfo imageViewInfo = {};
    imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.flags = 0;
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format = VK_FORMAT_D32_SFLOAT;
    imageViewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
    imageViewInfo.subresourceRange.baseMipLevel = 0;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.layerCount = 1;
    imageViewInfo.subresourceRange.levelCount = 1;
    imageViewInfo.image = VK_NULL_HANDLE;

    createImgAllocAndBind(a_device, a_physDevice, a_width, a_height, a_format, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                          &result, 
                          &imgCreateInfo, &imageViewInfo);

    return result;
  }


  void setImageLayout(VkCommandBuffer cmdBuffer,
                      VkImage image,
                      VkImageLayout oldImageLayout,
                      VkImageLayout newImageLayout,
                      VkImageSubresourceRange subresourceRange,
                      VkPipelineStageFlags srcStageMask,
                      VkPipelineStageFlags dstStageMask)
  {
    VkImageMemoryBarrier imageMemoryBarrier = {};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.oldLayout = oldImageLayout;
    imageMemoryBarrier.newLayout = newImageLayout;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = subresourceRange;

    // Source layouts (old)
    // Source access mask controls actions that have to be finished on the old layout
    // before it will be transitioned to the new layout
    switch (oldImageLayout)
    {
      case VK_IMAGE_LAYOUT_UNDEFINED:
        // Image layout is undefined (or does not matter)
        // Only valid as initial layout
        // No flags required, listed only for completeness
        imageMemoryBarrier.srcAccessMask = 0;
        break;

      case VK_IMAGE_LAYOUT_PREINITIALIZED:
        // Image is pre-initialized
        // Only valid as initial layout for linear images, preserves memory contents
        // Make sure host writes have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        // Image is a color attachment
        // Make sure any writes to the color buffer have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        // Image is a depth/stencil attachment
        // Make sure any writes to the depth/stencil buffer have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        // Image is a transfer source
        // Make sure any reads from the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;

      case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        // Image is a transfer destination
        // Make sure any writes to the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        // Image is read by a shader
        // Make sure any shader reads from the image have been finished
        imageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
        break;
      default:
        // Other source layouts aren't handled (yet)
        break;
    }

    // Target layouts
    // Destination access mask controls the dependency for the new image layout
    switch (newImageLayout)
    {
      case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
        // Image will be used as a transfer destination
        // Make sure any writes to the image have been finished
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
        // Image will be used as a transfer source
        // Make sure any reads from the image have been finished
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        break;

      case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
        // Image will be used as a color attachment
        // Make sure any writes to the color buffer have been finished
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
        // Image layout will be used as a depth/stencil attachment
        // Make sure any writes to depth/stencil buffer have been finished
        imageMemoryBarrier.dstAccessMask = imageMemoryBarrier.dstAccessMask | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        break;

      case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
        // Image will be read in a shader (sampler, input attachment)
        // Make sure any writes to the image have been finished
        if (imageMemoryBarrier.srcAccessMask == 0)
        {
          imageMemoryBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        }
        imageMemoryBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        break;
      default:
        // Other source layouts aren't handled (yet)
        break;
    }

    vkCmdPipelineBarrier(
        cmdBuffer,
        srcStageMask,
        dstStageMask,
        0,
        0, nullptr,
        0, nullptr,
        1, &imageMemoryBarrier);
  }

  // Fixed sub resource on first mip level and layer
  void setImageLayout(
      VkCommandBuffer cmdBuffer,
      VkImage image,
      VkImageAspectFlags aspectMask,
      VkImageLayout oldImageLayout,
      VkImageLayout newImageLayout,
      VkPipelineStageFlags srcStageMask,
      VkPipelineStageFlags dstStageMask)
  {
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = aspectMask;
    subresourceRange.baseMipLevel = 0;
    subresourceRange.levelCount = 1;
    subresourceRange.layerCount = 1;
    setImageLayout(cmdBuffer, image, oldImageLayout, newImageLayout, subresourceRange, srcStageMask, dstStageMask);
  }

  void insertImageMemoryBarrier(
      VkCommandBuffer cmdBuffer,
      VkImage image,
      VkAccessFlags srcAccessMask,
      VkAccessFlags dstAccessMask,
      VkImageLayout oldImageLayout,
      VkImageLayout newImageLayout,
      VkPipelineStageFlags srcStageMask,
      VkPipelineStageFlags dstStageMask,
      VkImageSubresourceRange subresourceRange)
  {
    VkImageMemoryBarrier imageMemoryBarrier {};
    imageMemoryBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imageMemoryBarrier.srcAccessMask = srcAccessMask;
    imageMemoryBarrier.dstAccessMask = dstAccessMask;
    imageMemoryBarrier.oldLayout = oldImageLayout;
    imageMemoryBarrier.newLayout = newImageLayout;
    imageMemoryBarrier.image = image;
    imageMemoryBarrier.subresourceRange = subresourceRange;

    vkCmdPipelineBarrier(
        cmdBuffer,
        srcStageMask,
        dstStageMask,
        0,
        0, nullptr,
        0, nullptr,
        1, &imageMemoryBarrier);
  }

  VkSampler createSampler(VkDevice a_device, VkFilter a_filterMode, VkSamplerAddressMode a_addressMode,
                          VkBorderColor a_border_color, uint32_t a_mipLevels)
  {
    VkSampler result;

    VkSamplerCreateInfo samplerCreateInfo = {};
    samplerCreateInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerCreateInfo.maxAnisotropy = 1.0f;
    samplerCreateInfo.magFilter = a_filterMode;
    samplerCreateInfo.minFilter = a_filterMode;
    samplerCreateInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerCreateInfo.addressModeU = a_addressMode;
    samplerCreateInfo.addressModeV = a_addressMode;
    samplerCreateInfo.addressModeW = a_addressMode;
    samplerCreateInfo.mipLodBias = 0.0f;
    samplerCreateInfo.compareOp = VK_COMPARE_OP_NEVER;
    samplerCreateInfo.minLod = 0.0f;
    samplerCreateInfo.maxLod = 0.0f;
    samplerCreateInfo.borderColor = a_border_color;
    samplerCreateInfo.maxAnisotropy = 1.0;
    samplerCreateInfo.anisotropyEnable = VK_FALSE;

    samplerCreateInfo.maxLod = (float)a_mipLevels;

    VK_CHECK_RESULT(vkCreateSampler(a_device, &samplerCreateInfo, nullptr, &result));

    return result;
  }

  void recordMipChainGenerationCmdBuf(VkDevice a_device, VkCommandBuffer a_cmdBuf, const VulkanImageMem& imageMem,
                                      uint32_t a_width, uint32_t a_height, uint32_t a_mipLevels, VkImageLayout a_targetLayout)
  {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkBeginCommandBuffer(a_cmdBuf, &beginInfo);
    // Copy down mips from n-1 to n
    for (uint32_t i = 1; i < a_mipLevels; i++) //TODO: rewrite with staging buffer and possibly move this to copy helper
    {
      VkImageBlit imageBlit{};

      // Source
      imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBlit.srcSubresource.layerCount = 1;
      imageBlit.srcSubresource.mipLevel = i - 1;
      imageBlit.srcOffsets[1].x = int32_t(a_width >> (i - 1));
      imageBlit.srcOffsets[1].y = int32_t(a_height >> (i - 1));
      imageBlit.srcOffsets[1].z = 1;

      // Destination
      imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBlit.dstSubresource.layerCount = 1;
      imageBlit.dstSubresource.mipLevel = i;
      imageBlit.dstOffsets[1].x = int32_t(a_width >> i);
      imageBlit.dstOffsets[1].y = int32_t(a_height >> i);
      imageBlit.dstOffsets[1].z = 1;

      VkImageSubresourceRange mipSubRange = {};
      mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      mipSubRange.baseMipLevel = i;
      mipSubRange.levelCount = 1;
      mipSubRange.layerCount = 1;

      // Transition current mip level to transfer dest
      vk_utils::setImageLayout(
          a_cmdBuf,
          imageMem.image,
          VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          mipSubRange,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT);

      // Blit from previous level
      vkCmdBlitImage(
          a_cmdBuf,
          imageMem.image,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          imageMem.image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          1,
          &imageBlit,
          VK_FILTER_LINEAR);

      // Transition current mip level to transfer source for read in next iteration
      vk_utils::setImageLayout(
          a_cmdBuf,
          imageMem.image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
          mipSubRange,
          VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    // After the loop, all mip layers are in TRANSFER_SRC layout, so transition all to SHADER_READ
    VkImageSubresourceRange subresourceRange = {};
    subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    subresourceRange.levelCount = a_mipLevels;
    subresourceRange.layerCount = 1;
    vk_utils::setImageLayout(
        a_cmdBuf,
        imageMem.image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        a_targetLayout,
        subresourceRange);

    vkEndCommandBuffer(a_cmdBuf);
  }
}