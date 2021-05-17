#ifndef VULKAN_FSQUAD_H
#define VULKAN_FSQUAD_H

#include "vk_utils.h"

namespace vkfw
{
  /**
  \brief This struction contain enough info for enabling render-to-texture in Vulkan and creating all additional Vulkan objects
  */
  struct RenderTargetInfo2D
  {
    VkExtent2D         size;           //!< image resolution
    VkFormat           fmt;            //!< image format 
    VkAttachmentLoadOp loadOp;         //!< information for renderpass
    VkImageLayout      initialLayout;  //!< information for renderpass
    VkImageLayout      finalLayout;    //!< information for renderpass
  };

  void CreateRenderPass(VkDevice a_device, RenderTargetInfo2D a_rtInfo,
                        VkRenderPass* a_pRenderPass);

  /**
  \brief simple helper for drawing textured quads (2D rectangles) on screen 
  */
  struct FSQuad
  {
    FSQuad() : m_pipeline(nullptr), m_layout(nullptr),  m_renderPass(nullptr), m_fbTarget(nullptr),
               m_dlayout(nullptr) {}

    virtual ~FSQuad();

    /**
    \brief Create resources that are needed to draw textured quad
    \param a_device - input Vulkan logical device
    \param a_vspath - input path to special quad vertex shader (compiled to SPIR-V)
    \param a_fspath - input path to quad fragment shader       (compiled to SPIR-V)
    \param a_rtInfo - input render target info; you shoud specify it due to Vulkan requires a lot of detailsto be specified. 
                      I. e. you should know a lot in advance about images that you are going to render in to. 

    */
    void Create(VkDevice a_device, const char* a_vspath, const char* a_fspath, RenderTargetInfo2D a_rtInfo);

    /**
    \brief This function allow you to set/change target image that you are going to render in to.
    \param - input image view of the output image

      The future implementations is assume to have some cache of vulkan frame buffer objects for each input image view
      The current implementation make new framebuffer for each call of 'SetRenderTarget' (destroying the old one of cource).
    */
    void SetRenderTarget(VkImageView a_imageView); 


    /**
    \brief Writes commands of drawing quad
    \param a_cmdBuff         - output command buffer. 
    \param a_inTexDescriptor - input descriptor set for texture that will be assigned to a quad; it is assumed that user will create descriptor for current ... 
    \param a_offsAndScale    - input array of packed scale ([0],[1]) and offset ([2],[3]);
   
    */
    void DrawCmd(VkCommandBuffer a_cmdBuff, VkDescriptorSet a_inTexDescriptor, float a_offsAndScale[4]);

  protected:

    FSQuad(const FSQuad& a_rhs) { }
    FSQuad& operator=(const FSQuad& a_rhs) { return *this; }

    VkDevice         m_device = {};
    VkPipeline       m_pipeline;
    VkPipelineLayout m_layout;
    VkRenderPass     m_renderPass;
    VkExtent2D       m_fbSize = {};

    VkFramebuffer    m_fbTarget;
    VkImageView      m_targetView = {};
    
    // this is for binding texture
    //
    VkDescriptorSetLayout m_dlayout;
    RenderTargetInfo2D    m_rtCreateInfo = {};

  };

  class QuadRenderer
  {
  public:
    QuadRenderer() : m_pipeline(nullptr), m_layout(nullptr), m_renderPass(nullptr), m_fbTarget(nullptr),
      m_dlayout(nullptr) {}
    ~QuadRenderer();

    void Create(VkDevice a_device, const char* a_vspath, const char* a_fspath, RenderTargetInfo2D a_rtInfo, VkRect2D scissor);
    void SetRenderTarget(VkImageView a_imageView);
    void DrawCmd(VkCommandBuffer a_cmdBuff, VkDescriptorSet a_inTexDescriptor);

  private:

    QuadRenderer(const QuadRenderer& a_rhs) = delete;
    QuadRenderer& operator=(const QuadRenderer& a_rhs) = delete;

    VkDevice         m_device = {};
    VkPipeline       m_pipeline;
    VkPipelineLayout m_layout;
    VkRenderPass     m_renderPass;
    VkExtent2D       m_fbSize = {};

    VkFramebuffer    m_fbTarget;
    VkImageView      m_targetView;

    // this is for binding texture
    //
    VkDescriptorSetLayout m_dlayout;
    RenderTargetInfo2D    m_rtCreateInfo = {};
    VkRect2D              rect;
  };

};

#endif
