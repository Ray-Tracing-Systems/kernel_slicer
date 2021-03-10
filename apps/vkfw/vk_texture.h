#ifndef VULKAN_TEXTURE_HELPER_H
#define VULKAN_TEXTURE_HELPER_H

#include "vulkan_wrapper/vulkan_include.h"
#include "vk_copy.h"

#include <vector>
#include <stdexcept>
#include <sstream>
#include <memory>

namespace vkfw
{
 
  struct ImageParameters
  {
    VkFormat format;
    uint32_t width;
    uint32_t height;
    uint32_t mipLevels;
    bool     filterable;
    bool     renderable;
    bool     transferable;
  };

  ImageParameters SimpleTextureParameters(VkFormat a_format, uint32_t a_width, uint32_t a_height);
  ImageParameters RTTextureParameters    (VkFormat a_format, uint32_t a_width, uint32_t a_height);

  struct BaseTexture2D
  {
    BaseTexture2D() : m_memStorage(0), m_image(0), m_sampler(0), m_view(0), m_device(0) {}
    ~BaseTexture2D();

    //Methods
    VkMemoryRequirements CreateImage(VkDevice a_device, const ImageParameters& a_params);
    VkMemoryRequirements CreateImage(VkDevice a_device, uint32_t a_width, uint32_t a_height, VkFormat a_format) 
    { 
      auto params = SimpleTextureParameters(a_format, a_width, a_height);
      if(a_format == VK_FORMAT_R32G32B32A32_UINT)
      {
        params.filterable = false;
        params.mipLevels  = 1;
      }
      return CreateImage(a_device, params); 
    }

    void                 BindMemory (VkDeviceMemory a_memStorage, size_t a_offset);
    void                 Update     (const void* a_src, int a_width, int a_height, int a_bpp, vkfw::ICopyEngine* a_pCopyImpl);

    void                 GenerateMipsCmd(VkCommandBuffer a_cmdBuff);
    void                 ChangeLayoutCmd(VkCommandBuffer a_cmdBuff, VkImageLayout a_newLayout, VkPipelineStageFlags a_newStage);

    //Getters
    VkImageView   View()    const { return m_view;  }
    VkImage       Image()   const { return m_image; }
    VkSampler     Sampler() const { return m_sampler; }
    VkDevice      Device()  const { return m_device; }
    VkImageLayout Layout()  const { return m_currentLayout; }

    uint32_t      Width()     const { return m_params.width; }
    uint32_t      Height()    const { return m_params.height; }
    uint32_t      MipsCount() const { return m_params.mipLevels; }
    VkFormat      Format()    const { return m_params.format; }

    //Setters
    void SetSampler(VkSampler     a_sampler) { vkDestroySampler(m_device, m_sampler, NULL); m_sampler = a_sampler; }

  private:
    VkDeviceMemory  m_memStorage; // SimpleVulkanTexture DOES NOT OWN memStorage! It just save reference to it.
    VkImage         m_image;
    VkSampler       m_sampler;
    VkImageView     m_view;
    VkDevice        m_device;
    ImageParameters m_params = {};

    VkImageCreateInfo    m_createImageInfo = {};
    VkImageLayout        m_currentLayout   = {};
    VkPipelineStageFlags m_currentStage    = {};

    friend struct RenderableTexture2D;
  };

  typedef BaseTexture2D SimpleTexture2D;

  struct RenderableTexture2D
  {
    RenderableTexture2D() : m_fbo(0), m_renderPass(0) { }
    ~RenderableTexture2D();

    //// useful functions
    //
    VkMemoryRequirements CreateImage(VkDevice a_device, const ImageParameters& a_params);
    VkMemoryRequirements CreateImage(VkDevice a_device, uint32_t a_width, uint32_t a_height, VkFormat a_format) { return CreateImage(a_device, RTTextureParameters(a_format, a_width, a_height)); }

    void                 BindMemory(VkDeviceMemory a_memStorage, size_t a_offset);
  
    VkImageView   View()    const { return m_imageData.m_view;  }
    VkImage       Image()   const { return m_imageData.m_image; }
    VkSampler     Sampler() const { return m_imageData.m_sampler; }
    VkDevice      Device()  const { return m_imageData.m_device; }
    VkImageLayout Layout()  const { return m_imageData.m_currentLayout; }

    uint32_t      Width()     const { return m_imageData.m_params.width; }
    uint32_t      Height()    const { return m_imageData.m_params.height; }
    uint32_t      MipsCount() const { return m_imageData.m_params.mipLevels; }
    VkFormat      Format()    const { return m_imageData.m_params.format; }


    void                 BeginRenderingToThisTexture(VkCommandBuffer a_cmdBuff);
    void                 EndRenderingToThisTexture(VkCommandBuffer a_cmdBuff);

    //// information functions
    //
    VkRenderPass          Renderpass()  const { return m_renderPass; }
    VkFramebuffer         Framebuffer() const { return m_fbo;  }

  protected:

    void CreateRenderPass();
    VkImageLayout RenderAttachmentLayout();

    VkRenderPass   m_renderPass;
    VkFramebuffer  m_fbo;

    BaseTexture2D m_imageData;
  };

};

#endif