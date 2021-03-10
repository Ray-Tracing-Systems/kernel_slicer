#pragma once

#include <map>
#include <unordered_map>
#include "vk_utils.h"

#if defined(__ANDROID__)
#include <android_native_app_glue.h>
#endif

namespace vkfw
{
  VkPipelineInputAssemblyStateCreateInfo IA_TList();
  VkPipelineInputAssemblyStateCreateInfo IA_PList();
  VkPipelineInputAssemblyStateCreateInfo IA_LList();
  VkPipelineInputAssemblyStateCreateInfo IA_LSList();

  struct GraphicsPipelineMaker
  {
    enum {MAXSHADERS = 10};
    VkShaderModule                  shaderModules[MAXSHADERS];
    VkPipelineShaderStageCreateInfo shaderStageInfos[MAXSHADERS];

    VkPipelineInputAssemblyStateCreateInfo inputAssembly;                  // set from input!
    VkViewport                             viewport;
    VkRect2D                               scissor;
    VkPipelineViewportStateCreateInfo      viewportState;
    VkPipelineRasterizationStateCreateInfo rasterizer;
    VkPipelineMultisampleStateCreateInfo   multisampling;
    VkPipelineColorBlendAttachmentState    colorBlendAttachment;
    VkPipelineColorBlendStateCreateInfo    colorBlending;
    VkPushConstantRange                    pcRange;
    VkPipelineLayoutCreateInfo             pipelineLayoutInfo;
    VkPipelineDepthStencilStateCreateInfo  depthStencilTest;
    VkGraphicsPipelineCreateInfo           pipelineInfo;

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    GraphicsPipelineMaker();
#if defined(__ANDROID__)
    void             CreateShaders(AAssetManager* mgr, VkDevice a_device, const std::unordered_map<VkShaderStageFlagBits, std::string>& shader_paths);
#else
    void             CreateShaders(VkDevice a_device, const std::unordered_map<VkShaderStageFlagBits, std::string> &shader_paths);
#endif
    void             SetDefault_Simple3D(uint32_t a_width, uint32_t a_height);

    VkPipelineLayout MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout, uint32_t a_pcRangeSize);
    VkPipeline       MakePipeline(VkDevice a_device, VkPipelineVertexInputStateCreateInfo a_vertexLayout, VkRenderPass a_renderPass,
                                  VkPipelineInputAssemblyStateCreateInfo a_inputAssembly = IA_TList());

  protected:

    VkPipelineLayout m_pipelineLayout;
    VkPipeline       m_pipeline;
    int              m_stagesNum;
  };

}
