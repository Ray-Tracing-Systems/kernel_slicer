#include <cassert>
#include <array>
#include "vk_fbuf_attachment.h"
#include "vk_utils.h"
#include "vk_images.h"

namespace vk_utils
{

  RenderTarget::~RenderTarget()
  {
    assert(m_device);
    for (auto &attachment : m_attachments)
    {
      vkDestroyImage(m_device, attachment.image, nullptr);
      vkDestroyImageView(m_device, attachment.view, nullptr);
    }

    vkDestroySampler(m_device, m_sampler, nullptr);
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);
    for (auto &fbuf : m_framebuffers)
    {
      vkDestroyFramebuffer(m_device, fbuf, nullptr);
    }

  }

  uint32_t RenderTarget::CreateAttachment(const AttachmentInfo &a_info)
  {
    FbufAttachment attachment;
    attachment.format = a_info.format;

    VkImageAspectFlags aspectMask = 0;

    if (a_info.usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    {
      aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    }

    if (a_info.usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    {
      if (vk_utils::isDepthFormat(attachment.format))
      {
        aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
      }
      if (vk_utils::isStencilFormat(attachment.format))
      {
        aspectMask = aspectMask | VK_IMAGE_ASPECT_STENCIL_BIT;
      }
    }

    assert(aspectMask > 0);

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = m_resolution.width;
    imageInfo.extent.height = m_resolution.height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = a_info.format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = a_info.usage;
    imageInfo.samples = a_info.imageSampleCount;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateImage(m_device, &imageInfo, nullptr, &attachment.image));
    vkGetImageMemoryRequirements(m_device, attachment.image, &attachment.mem_req);

    attachment.layerCount = 1;

    attachment.subresourceRange = {};
    attachment.subresourceRange.aspectMask = aspectMask;
    attachment.subresourceRange.levelCount = 1;
    attachment.subresourceRange.layerCount = 1;

    attachment.description = {};
    attachment.description.samples = a_info.imageSampleCount;
    attachment.description.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.description.storeOp = (a_info.usage & VK_IMAGE_USAGE_SAMPLED_BIT) ? VK_ATTACHMENT_STORE_OP_STORE
                                                                                 : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.description.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.description.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.description.format = a_info.format;
    attachment.description.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.description.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    if (!vk_utils::isDepthFormat(a_info.format) && !vk_utils::isStencilFormat(a_info.format))
    {
      m_numColorAttachments++;
    }

    m_attachments.push_back(attachment);

    return static_cast<uint32_t>(m_attachments.size() - 1);
  }

  void RenderTarget::CreateViewAndBindMemory(VkDeviceMemory a_mem, const std::vector <VkDeviceSize> &a_offsets)
  {
    assert(a_offsets.size() <= m_attachments.size());
    for (size_t i = 0; i < m_attachments.size(); ++i)
    {
      VK_CHECK_RESULT(vkBindImageMemory(m_device, m_attachments[i].image, a_mem, a_offsets[i]));
      m_attachments[i].mem = a_mem;
      m_attachments[i].mem_offset = a_offsets[i];

      VkImageViewCreateInfo imageView{};
      imageView.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      imageView.viewType = (m_attachments[i].layerCount == 1) ? VK_IMAGE_VIEW_TYPE_2D : VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      imageView.format = m_attachments[i].description.format;
      imageView.subresourceRange = m_attachments[i].subresourceRange;

      imageView.image = m_attachments[i].image;
      VK_CHECK_RESULT(vkCreateImageView(m_device, &imageView, nullptr, &m_attachments[i].view));
    }
  }

  VkResult RenderTarget::CreateDefaultSampler()
  {
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = 1.0f;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    return vkCreateSampler(m_device, &samplerInfo, nullptr, &m_sampler);
  }

  VkResult RenderTarget::CreateDefaultRenderPass()
  {
    std::vector <VkAttachmentDescription> attachmentDescriptions;
    for (auto &attachment : m_attachments)
    {
      attachmentDescriptions.push_back(attachment.description);
    };

    std::vector <VkAttachmentReference> colorReferences;
    VkAttachmentReference depthReference = {};

    bool hasDepth = false;
    bool hasColor = false;
    for (uint32_t i = 0; i < m_attachments.size(); ++i)
    {
      if (vk_utils::isDepthOrStencil(m_attachments[i].format))
      {
        assert(!hasDepth);
        depthReference.attachment = i;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        hasDepth = true;
      }
      else
      {
        colorReferences.push_back({i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        hasColor = true;
      }
    }

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (hasColor)
    {
      subpass.pColorAttachments = colorReferences.data();
      subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
    }
    if (hasDepth)
    {
      subpass.pDepthStencilAttachment = &depthReference;
    }

    std::array<VkSubpassDependency, 2> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.pAttachments = attachmentDescriptions.data();
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescriptions.size());
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 2;
    renderPassInfo.pDependencies = dependencies.data();
    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass));

    std::vector <VkImageView> attachmentViews;
    for (auto attachment : m_attachments)
    {
      attachmentViews.push_back(attachment.view);
    }

    uint32_t maxLayers = 0;
    for (auto attachment : m_attachments)
    {
      if (attachment.subresourceRange.layerCount > maxLayers)
      {
        maxLayers = attachment.subresourceRange.layerCount;
      }
    }

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass = m_renderPass;
    framebufferInfo.pAttachments = attachmentViews.data();
    framebufferInfo.attachmentCount = static_cast<uint32_t>(attachmentViews.size());
    framebufferInfo.width = m_resolution.width;
    framebufferInfo.height = m_resolution.height;
    framebufferInfo.layers = maxLayers;
    VkFramebuffer fbuf;
    VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &fbuf));
    m_framebuffers.push_back(fbuf);

    return VK_SUCCESS;
  }

  VkRenderPassBeginInfo
  RenderTarget::GetRenderPassBeginInfo(uint32_t a_fbufIdx, std::vector <VkClearValue> &a_clearValues,
                                       VkOffset2D a_renderOffset) const
  {
    if (a_clearValues.size() != m_attachments.size())
    {
      vk_utils::logWarning("[RenderTarget::GetRenderPassBeginInfo] clear values size doesn't match attachment count");
    }

    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_renderPass;
    renderPassInfo.framebuffer = m_framebuffers[a_fbufIdx];
    renderPassInfo.renderArea.offset = a_renderOffset;
    renderPassInfo.renderArea.extent = m_resolution;
    renderPassInfo.clearValueCount = a_clearValues.size();
    renderPassInfo.pClearValues = a_clearValues.data();

    return renderPassInfo;
  }

  std::vector <VkMemoryRequirements> RenderTarget::GetMemoryRequirements() const
  {
    std::vector <VkMemoryRequirements> result;
    for (size_t i = 0; i < m_attachments.size(); ++i)
    {
      result.push_back(m_attachments[i].mem_req);
    }

    return result;
  }

  VkResult RenderTarget::CreateRenderPassWithSwapchainOut(VkFormat a_swapChainFormat,
                                                          const std::vector <VkImageView> &a_swapChainImageViews)
  {
    std::vector <VkAttachmentDescription> attachmentDescriptions;
    for (auto &attachment : m_attachments)
    {
      attachmentDescriptions.push_back(attachment.description);
    };

    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format = a_swapChainFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    attachmentDescriptions.push_back(colorAttachment);

    std::vector <VkAttachmentReference> colorReferences;
    VkAttachmentReference depthReference = {};

    bool hasDepth = false;
    bool hasColor = false;
    for (uint32_t i = 0; i < m_attachments.size(); ++i)
    {
      if (vk_utils::isDepthOrStencil(m_attachments[i].format))
      {
        assert(!hasDepth);
        depthReference.attachment = i;
        depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        hasDepth = true;
      }
      else
      {
        colorReferences.push_back({i, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});
        hasColor = true;
      }
    }

    colorReferences.push_back({static_cast<uint32_t>(m_attachments.size()), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL});

    VkSubpassDescription subpass = {};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    if (hasColor)
    {
      subpass.pColorAttachments = colorReferences.data();
      subpass.colorAttachmentCount = static_cast<uint32_t>(colorReferences.size());
    }
    if (hasDepth)
    {
      subpass.pDepthStencilAttachment = &depthReference;
    }

    std::array<VkSubpassDependency, 2> dependencies{};

    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                   VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                   VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_INPUT_ATTACHMENT_READ_BIT;
    dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.pAttachments = attachmentDescriptions.data();
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachmentDescriptions.size());
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 2;
    renderPassInfo.pDependencies = dependencies.data();
    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_renderPass));


    uint32_t maxLayers = 0;
    for (auto attachment : m_attachments)
    {
      if (attachment.subresourceRange.layerCount > maxLayers)
      {
        maxLayers = attachment.subresourceRange.layerCount;
      }
    }

    for (size_t i = 0; i < a_swapChainImageViews.size(); i++)
    {
      std::vector <VkImageView> attachmentViews;
      for (auto attachment : m_attachments)
      {
        attachmentViews.push_back(attachment.view);
      }
      attachmentViews.push_back(a_swapChainImageViews[i]);

      VkFramebufferCreateInfo framebufferInfo = {};
      framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      framebufferInfo.renderPass = m_renderPass;
      framebufferInfo.pAttachments = attachmentViews.data();
      framebufferInfo.attachmentCount = static_cast<uint32_t>(attachmentViews.size());
      framebufferInfo.width = m_resolution.width;
      framebufferInfo.height = m_resolution.height;
      framebufferInfo.layers = maxLayers;
      VkFramebuffer fbuf;
      VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &fbuf));
      m_framebuffers.push_back(fbuf);
    }

    return VK_SUCCESS;
  }
}