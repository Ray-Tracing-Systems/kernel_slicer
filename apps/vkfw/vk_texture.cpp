#include "vk_texture.h"
#include "vk_utils.h"
#include <cassert>
#include <cmath>
#include <algorithm>

#ifdef WIN32
#undef max
#undef min
#endif

static size_t Padding(size_t a_size, size_t a_aligment)
{
  if (a_size % a_aligment == 0)
    return a_size;
  else
  {
    size_t sizeCut = a_size - (a_size % a_aligment);
    return sizeCut + a_aligment;
  }
}

static bool CheckFilterable(const vkfw::ImageParameters& a_params)
{
  return a_params.format != VK_FORMAT_R32G32B32A32_UINT;
}

static VkImageUsageFlags GetTextureUsage(const vkfw::ImageParameters& a_params)
{
  VkImageUsageFlags usage = VK_IMAGE_USAGE_SAMPLED_BIT;
  if (a_params.transferable)
    usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT; // copy to the texture and read then
  if (a_params.renderable)
  {
    const bool isDepthTexture = vk_utils::IsDepthFormat(a_params.format);
    const auto attachmentFlags = isDepthTexture ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    usage |= attachmentFlags;
  }
  return usage;
}

static void BindImageToMemoryAndCreateImageView(VkDevice a_device, VkImage a_image, VkFormat a_format, uint32_t a_mipLevels,
  VkDeviceMemory a_memStorage, size_t a_offset,
  VkImageView* a_pView)
{
  VK_CHECK_RESULT(vkBindImageMemory(a_device, a_image, a_memStorage, a_offset));

  const bool isDepthTexture = vk_utils::IsDepthFormat(a_format);

  VkImageViewCreateInfo imageViewInfo = {};
  {
    imageViewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    imageViewInfo.flags = 0;
    imageViewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    imageViewInfo.format = a_format;
    imageViewInfo.components = { VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G, VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A };
    imageViewInfo.subresourceRange.aspectMask = isDepthTexture ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    imageViewInfo.subresourceRange.baseMipLevel = 0;
    imageViewInfo.subresourceRange.baseArrayLayer = 0;
    imageViewInfo.subresourceRange.layerCount = 1;
    imageViewInfo.subresourceRange.levelCount = a_mipLevels;
    imageViewInfo.image = a_image;     // The view will be based on the texture's image
  }
  VK_CHECK_RESULT(vkCreateImageView(a_device, &imageViewInfo, nullptr, a_pView));
}



namespace vkfw
{

  ImageParameters SimpleTextureParameters(VkFormat a_format, uint32_t a_width, uint32_t a_height)
  {
    ImageParameters parameters;
    parameters.format       = a_format;
    parameters.width        = a_width;
    parameters.height       = a_height;
    parameters.mipLevels    = int(floor(log2(std::max(parameters.width, parameters.height) + 1)));
    parameters.filterable   = true;
    parameters.renderable   = false;
    parameters.transferable = true;
    return parameters;
  }

  ImageParameters RTTextureParameters(VkFormat a_format, uint32_t a_width, uint32_t a_height)
  {
    ImageParameters parameters;
    parameters.format       = a_format;
    parameters.width        = a_width;
    parameters.height       = a_height;
    parameters.mipLevels    = 1;
    parameters.filterable   = false;
    parameters.renderable   = true;
    parameters.transferable = false;
    return parameters;
  }

  BaseTexture2D::~BaseTexture2D()
  {
    if (m_device == nullptr)
      return;

    vkDestroyImage(m_device, m_image, NULL);   m_image = nullptr;
    vkDestroyImageView(m_device, m_view, NULL);    m_view = nullptr;
    vkDestroySampler(m_device, m_sampler, NULL); m_sampler = nullptr;

    m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    m_currentStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }

  VkMemoryRequirements BaseTexture2D::CreateImage(VkDevice a_device, const ImageParameters& a_params)
  {
    assert(a_params.format != VK_FORMAT_R32G32B32A32_UINT || a_params.mipLevels == 1); // Do we really need this assert?
    assert(CheckFilterable(a_params) || !a_params.filterable);
    m_device = a_device;
    m_params = a_params;

    VkImageCreateInfo imgCreateInfo = {};
    imgCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imgCreateInfo.pNext = nullptr;
    imgCreateInfo.flags = 0; // not sure about this ...
    imgCreateInfo.imageType = VK_IMAGE_TYPE_2D;
    imgCreateInfo.format = a_params.format;
    imgCreateInfo.extent = VkExtent3D{ uint32_t(a_params.width), uint32_t(a_params.height), 1 };
    imgCreateInfo.mipLevels = m_params.mipLevels;
    imgCreateInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgCreateInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgCreateInfo.usage = GetTextureUsage(a_params);
    imgCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgCreateInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgCreateInfo.arrayLayers = 1;

    VK_CHECK_RESULT(vkCreateImage(a_device, &imgCreateInfo, nullptr, &(m_image)));
    m_currentLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VkMemoryRequirements memoryRequirements;
    vkGetImageMemoryRequirements(a_device, m_image, &memoryRequirements);

    VkSamplerCreateInfo samplerInfo = {};
    {
      samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      samplerInfo.pNext = nullptr;
      samplerInfo.flags = 0;
      samplerInfo.magFilter = a_params.filterable ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
      samplerInfo.minFilter = a_params.filterable ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
      samplerInfo.mipmapMode = a_params.filterable ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
      samplerInfo.addressModeU = a_params.filterable ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeV = a_params.filterable ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.addressModeW = a_params.filterable ? VK_SAMPLER_ADDRESS_MODE_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      samplerInfo.mipLodBias = 0.0f;
      samplerInfo.compareOp = VK_COMPARE_OP_NEVER;
      samplerInfo.minLod = 0;
      samplerInfo.maxLod = float(m_params.mipLevels);
      samplerInfo.maxAnisotropy = 1.0;
      samplerInfo.anisotropyEnable = VK_FALSE;
      samplerInfo.borderColor = vk_utils::IsDepthFormat(a_params.format) ? VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK : VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      samplerInfo.unnormalizedCoordinates = VK_FALSE;
    }
    VK_CHECK_RESULT(vkCreateSampler(a_device, &samplerInfo, nullptr, &m_sampler));

    m_createImageInfo = imgCreateInfo;
    memoryRequirements.size = Padding(memoryRequirements.size, memoryRequirements.alignment * 4);
    return memoryRequirements;
  }

  void BaseTexture2D::BindMemory(VkDeviceMemory a_memStorage, size_t a_offset)
  {
    assert(m_memStorage == nullptr); // this implementation does not allow to rebind memory!
    assert(m_view == nullptr); // may be later ...

    m_memStorage = a_memStorage;

    BindImageToMemoryAndCreateImageView(m_device, m_image, m_params.format, m_params.mipLevels,
                                        a_memStorage, a_offset,
                                        &(m_view));
  }

  void BaseTexture2D::Update(const void* a_src, int a_width, int a_height, int a_bpp, vkfw::ICopyEngine* a_pCopyImpl)
  {
    assert(a_pCopyImpl != nullptr);

    a_pCopyImpl->UpdateImage(m_image, a_src, a_width, a_height, a_bpp);

    m_currentLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    m_currentStage  = VK_PIPELINE_STAGE_TRANSFER_BIT;
  }
}

namespace vk_utils
{
    static void setImageLayout(
            VkCommandBuffer cmdbuffer,
            VkImage image,
            VkImageLayout oldImageLayout,
            VkImageLayout newImageLayout,
            VkImageSubresourceRange subresourceRange,
            VkPipelineStageFlags srcStageMask,
            VkPipelineStageFlags dstStageMask)
    {
      // Create an image barrier object
      VkImageMemoryBarrier imageMemoryBarrier = {};
      imageMemoryBarrier.sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imageMemoryBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imageMemoryBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imageMemoryBarrier.oldLayout           = oldImageLayout;
      imageMemoryBarrier.newLayout           = newImageLayout;
      imageMemoryBarrier.image               = image;
      imageMemoryBarrier.subresourceRange    = subresourceRange;

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
          // Image is preinitialized
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

      // Target layouts (new)
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

      // Put barrier inside setup command buffer
      vkCmdPipelineBarrier(
              cmdbuffer,
              srcStageMask,
              dstStageMask,
              0,
              0, nullptr,
              0, nullptr,
              1, &imageMemoryBarrier);
    }
};

namespace vkfw
{

  void BaseTexture2D::GenerateMipsCmd(VkCommandBuffer a_cmdBuff)
  {
    VkCommandBuffer blitCmd = a_cmdBuff;

    // at first, transfer 0 mip level to the VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    {
      VkImageMemoryBarrier imgBar = {};

      imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imgBar.pNext = nullptr;
      imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      imgBar.srcAccessMask = 0;
      imgBar.dstAccessMask = 0;
      imgBar.oldLayout = m_currentLayout;
      imgBar.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      imgBar.image = m_image;

      imgBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imgBar.subresourceRange.baseMipLevel = 0;
      imgBar.subresourceRange.levelCount = 1;
      imgBar.subresourceRange.baseArrayLayer = 0;
      imgBar.subresourceRange.layerCount = 1;

      vkCmdPipelineBarrier(blitCmd,
        m_currentStage,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &imgBar);
    }

    // Copy down mips from n-1 to n
    for (int32_t i = 1; i < m_params.mipLevels; i++)
    {
      VkImageBlit imageBlit{};

      // Source
      imageBlit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBlit.srcSubresource.layerCount = 1;
      imageBlit.srcSubresource.mipLevel = i - 1;
      imageBlit.srcOffsets[1].x = int32_t(m_params.width >> (i - 1));
      imageBlit.srcOffsets[1].y = int32_t(m_params.height >> (i - 1));
      imageBlit.srcOffsets[1].z = 1;

      // Destination
      imageBlit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imageBlit.dstSubresource.layerCount = 1;
      imageBlit.dstSubresource.mipLevel = i;
      imageBlit.dstOffsets[1].x = int32_t(m_params.width >> i);
      imageBlit.dstOffsets[1].y = int32_t(m_params.height >> i);
      imageBlit.dstOffsets[1].z = 1;

      VkImageSubresourceRange mipSubRange = {};
      mipSubRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      mipSubRange.baseMipLevel = i;
      mipSubRange.levelCount = 1;
      mipSubRange.layerCount = 1;

      // Transition current mip level to transfer dest
      vk_utils::setImageLayout(
        blitCmd,
        m_image,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        mipSubRange,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);

      // Blit from previous level
      vkCmdBlitImage(
        blitCmd,
        m_image,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        m_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1,
        &imageBlit,
        VK_FILTER_LINEAR);

      // Transition current mip level to transfer source for read in next iteration
      vk_utils::setImageLayout(
        blitCmd,
        m_image,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        mipSubRange,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT);
    }

    // After the loop, all mip layers are in TRANSFER_SRC layout, so transition all to SHADER_READ
    {
      VkImageMemoryBarrier imgBar = {};

      imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imgBar.pNext = nullptr;
      imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

      imgBar.srcAccessMask = 0;
      imgBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      imgBar.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      imgBar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imgBar.image = m_image;

      imgBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      imgBar.subresourceRange.baseMipLevel = 0;
      imgBar.subresourceRange.levelCount = m_params.mipLevels;
      imgBar.subresourceRange.baseArrayLayer = 0;
      imgBar.subresourceRange.layerCount = 1;

      vkCmdPipelineBarrier(blitCmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &imgBar);
    }

    m_currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    m_currentStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;

  }

  void BaseTexture2D::ChangeLayoutCmd(VkCommandBuffer a_cmdBuff, VkImageLayout a_newLayout, VkPipelineStageFlags a_newStage)
  {
    if (m_currentLayout == a_newLayout && m_currentStage == a_newStage)
      return;

    VkImageMemoryBarrier imgBar = {};

    imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBar.pNext = nullptr;
    imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBar.srcAccessMask = 0;                                        // #NOTE: THIS IS NOT CORRECT! please use vk_utils::setImageLayout!
    imgBar.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;                // #NOTE: THIS IS NOT CORRECT! please use vk_utils::setImageLayout!
    imgBar.oldLayout = m_currentLayout;
    imgBar.newLayout = a_newLayout;
    imgBar.image = m_image;

    imgBar.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    imgBar.subresourceRange.baseMipLevel = 0;
    imgBar.subresourceRange.levelCount = m_params.mipLevels;
    imgBar.subresourceRange.baseArrayLayer = 0;
    imgBar.subresourceRange.layerCount = 1;

    vkCmdPipelineBarrier(a_cmdBuff,
      m_currentStage,
      a_newStage,
      0,
      0, nullptr,
      0, nullptr,
      1, &imgBar);

    m_currentLayout = a_newLayout;
    m_currentStage = a_newStage;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  RenderableTexture2D::~RenderableTexture2D()
  {
    if (m_imageData.Device() == nullptr)
      return;

    vkDestroyRenderPass(m_imageData.Device(), m_renderPass, nullptr);
    vkDestroyFramebuffer(m_imageData.Device(), m_fbo, NULL);
  }

  VkMemoryRequirements RenderableTexture2D::CreateImage(VkDevice a_device, const ImageParameters& a_params)
  {
    return m_imageData.CreateImage(a_device, a_params);
  }

  VkImageLayout RenderableTexture2D::RenderAttachmentLayout()
  {
    const bool isDepthTexture = vk_utils::IsDepthFormat(m_imageData.Format());
    return isDepthTexture ? VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  }


  void RenderableTexture2D::CreateRenderPass()
  {
    const bool isDepthTexture = vk_utils::IsDepthFormat(m_imageData.Format());

    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = m_imageData.Format();
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = this->RenderAttachmentLayout();

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;

    if (isDepthTexture)
      subpass.pDepthStencilAttachment = &colorAttachmentRef;
    else
    {
      subpass.colorAttachmentCount = 1;
      subpass.pColorAttachments = &colorAttachmentRef;
    }

    VkSubpassDependency dependency = {};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if (vkCreateRenderPass(m_imageData.Device(), &renderPassInfo, nullptr, &m_renderPass) != VK_SUCCESS)
      throw std::runtime_error("[CreateRenderPass]: failed to create render pass!");
  }

  void RenderableTexture2D::BindMemory(VkDeviceMemory a_memStorage, size_t a_offset)
  {
    m_imageData.BindMemory(a_memStorage, a_offset);

    this->CreateRenderPass();

    // frame buffer objects
    //
    VkImageView attachments[] = { m_imageData.View() };

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_renderPass;
    framebufferInfo.attachmentCount = 1;
    framebufferInfo.pAttachments = attachments;
    framebufferInfo.width = m_imageData.Width();
    framebufferInfo.height = m_imageData.Height();
    framebufferInfo.layers = 1;

    if (vkCreateFramebuffer(m_imageData.Device(), &framebufferInfo, nullptr, &m_fbo) != VK_SUCCESS)
      throw std::runtime_error("[RenderableTexture2D]: failed to create framebuffer!");
  }

  void RenderableTexture2D::BeginRenderingToThisTexture(VkCommandBuffer a_cmdBuff)
  {
    const bool isDepthTexture = vk_utils::IsDepthFormat(m_imageData.Format());

    if (m_imageData.Layout() != VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) // we assume the texture is always should be in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL. render pass will use different intermediate layout.
    {
      VkImageMemoryBarrier imgBar = {};

      imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      imgBar.pNext = nullptr;
      imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      imgBar.srcAccessMask = 0;
      imgBar.dstAccessMask = 0; // VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
      imgBar.oldLayout = m_imageData.Layout();
      imgBar.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      imgBar.image = m_imageData.Image();

      imgBar.subresourceRange.aspectMask = isDepthTexture ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
      imgBar.subresourceRange.baseMipLevel = 0;
      imgBar.subresourceRange.levelCount = m_imageData.MipsCount();
      imgBar.subresourceRange.baseArrayLayer = 0;
      imgBar.subresourceRange.layerCount = 1;

      vkCmdPipelineBarrier(a_cmdBuff,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0,
        0, nullptr,
        0, nullptr,
        1, &imgBar);

      m_imageData.m_currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    }

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = this->Renderpass();
    renderPassInfo.framebuffer = this->Framebuffer();
    renderPassInfo.renderArea.offset = { 0, 0 };
    renderPassInfo.renderArea.extent = VkExtent2D{ m_imageData.Width(), m_imageData.Height() };

    VkClearValue clearValues[2] = {};

    if (isDepthTexture)
      clearValues[0].depthStencil = { 1.0f, 0 };
    else
      clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };

    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearValues[0];

    vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
  }

  void RenderableTexture2D::EndRenderingToThisTexture(VkCommandBuffer a_cmdBuff)
  {
    vkCmdEndRenderPass(a_cmdBuff);
    m_imageData.m_currentLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
  }

}
