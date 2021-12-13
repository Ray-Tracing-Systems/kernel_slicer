#include "vk_quad.h"
#include "vk_utils.h"

#include <vector>

#include <stdexcept>
#include <sstream>
#include <memory>


vk_utils::FSQuad::~FSQuad()
{
  if(m_pipeline != nullptr)
  {
    vkDestroyPipeline      (m_device, m_pipeline,   nullptr);
    vkDestroyPipelineLayout(m_device, m_layout,     nullptr);
    vkDestroyRenderPass    (m_device, m_renderPass, nullptr);
  }

  if(m_fbTarget != nullptr)
    vkDestroyFramebuffer(m_device, m_fbTarget, NULL);

  if(m_dlayout != nullptr)
    vkDestroyDescriptorSetLayout(m_device, m_dlayout, NULL);
}

void vk_utils::FSQuad::Create(VkDevice a_device, const char* a_vspath, const char* a_fspath, vk_utils::RenderTargetInfo2D a_rtInfo)
{
  m_device       = a_device;
  m_fbSize       = a_rtInfo.size;
  m_rtCreateInfo = a_rtInfo;
  
  auto vertShaderCode = vk_utils::readSPVFile(a_vspath);
  auto fragShaderCode = vk_utils::readSPVFile(a_fspath);
  
  if(vertShaderCode.size() == 0 || fragShaderCode.size() == 0)
    RUN_TIME_ERROR("[FSQuad::Create]: can not load shaders");

  VkShaderModule vertShaderModule = vk_utils::createShaderModule(a_device, vertShaderCode);
  VkShaderModule fragShaderModule = vk_utils::createShaderModule(a_device, fragShaderCode);

  // create pipeline layout first
  //
  VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
  vertShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage  = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName  = "main";  
  
  VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
  fragShaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName  = "main";  

  VkPipelineShaderStageCreateInfo      shaderStages[]  = { vertShaderStageInfo, fragShaderStageInfo };  
  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount   = 0;
  vertexInputInfo.vertexAttributeDescriptionCount = 0;  
  
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  inputAssembly.primitiveRestartEnable = VK_FALSE;  
  
  VkViewport viewport = {};
  viewport.x          = 0.0f;
  viewport.y          = 0.0f;
  viewport.width      = (float)m_fbSize.width;
  viewport.height     = (float)m_fbSize.height;
  viewport.minDepth   = 0.0f;
  viewport.maxDepth   = 1.0f;  
  
  VkRect2D scissor = {};
  scissor.offset   = { 0, 0 };
  scissor.extent   = m_fbSize;
  
  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports    = &viewport;
  viewportState.scissorCount  = 1;
  viewportState.pScissors     = &scissor;  

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable        = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode             = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth               = 1.0f;
  rasterizer.cullMode                = VK_CULL_MODE_NONE; // VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable         = VK_FALSE;  

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable  = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;  


  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable    = VK_FALSE;  

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable     = VK_FALSE;
  colorBlending.logicOp           = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount   = 1;
  colorBlending.pAttachments      = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;  

  // create ds layout for binding texture shader 
  //
  {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[1];
    
    descriptorSetLayoutBinding[0].binding            = 0;
    descriptorSetLayoutBinding[0].descriptorType     = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorSetLayoutBinding[0].descriptorCount    = 1;
    descriptorSetLayoutBinding[0].stageFlags         = VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorSetLayoutBinding[0].pImmutableSamplers = nullptr;  
  
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 1;
    descriptorSetLayoutCreateInfo.pBindings    = descriptorSetLayoutBinding;  
  
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &m_dlayout));
  }

  VkPushConstantRange pcRange = {};   
  pcRange.stageFlags = (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  pcRange.offset     = 0;
  pcRange.size       = 8*sizeof(float);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount         = 0;
  pipelineLayoutInfo.pushConstantRangeCount = 1; 
  pipelineLayoutInfo.pPushConstantRanges    = &pcRange;
  pipelineLayoutInfo.pSetLayouts            = &m_dlayout;
  pipelineLayoutInfo.setLayoutCount         = 1;

  VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, &m_layout));

  m_renderPass = vk_utils::createRenderPass(m_device, a_rtInfo);
  
  // finally create graphics pipeline
  //
  VkGraphicsPipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount          = 2;
  pipelineInfo.pStages             = shaderStages;
  pipelineInfo.pVertexInputState   = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState      = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState   = &multisampling;
  pipelineInfo.pColorBlendState    = &colorBlending;
  pipelineInfo.layout              = m_layout;
  pipelineInfo.renderPass          = m_renderPass;
  pipelineInfo.subpass             = 0;
  pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;  
  
  VK_CHECK_RESULT(vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

  vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
  vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
}

void vk_utils::FSQuad::SetRenderTarget(VkImageView a_imageView)
{
  if(m_fbTarget != nullptr)
    vkDestroyFramebuffer(m_device, m_fbTarget, NULL);

  //#TODO: add framebuffer cache inside this class ... 

  VkImageView attachments[] = { a_imageView };

  VkFramebufferCreateInfo framebufferInfo = {};
  framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass      = m_renderPass;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments    = attachments;
  framebufferInfo.width           = m_rtCreateInfo.size.width;
  framebufferInfo.height          = m_rtCreateInfo.size.height;
  framebufferInfo.layers          = 1;  

  VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_fbTarget));

  m_targetView = a_imageView; 
}

void vk_utils::FSQuad::DrawCmd(VkCommandBuffer a_cmdBuff, VkDescriptorSet a_inTexDescriptor, float a_offsAndScale[4])
{
  VkRenderPassBeginInfo renderPassInfo = {};
  renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass        = m_renderPass;
  renderPassInfo.framebuffer       = m_fbTarget;
  renderPassInfo.renderArea.offset = { 0, 0 };
  renderPassInfo.renderArea.extent = m_fbSize;

  VkClearValue clearValues[1]    = {};
  clearValues[0].color           = {0.0f, 0.0f, 0.0f, 1.0f};
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues    = &clearValues[0];  

  vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);  

  vkCmdBindPipeline      (a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &a_inTexDescriptor, 0, NULL);

  float scaleAndOffset[8] = {1.0f, 1.0f, 0.0f, 0.0f, 
                             0.0f, 1e6f, 1.0f, 0.0f};
  if(a_offsAndScale != 0)
  {
    scaleAndOffset[0] = a_offsAndScale[0];
    scaleAndOffset[1] = a_offsAndScale[1];
    scaleAndOffset[2] = a_offsAndScale[2];
    scaleAndOffset[3] = a_offsAndScale[3];
  }

  vkCmdPushConstants(a_cmdBuff, m_layout, (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT), 0, 8*sizeof(float), scaleAndOffset);

  vkCmdDraw(a_cmdBuff, 4, 1, 0, 0);

  vkCmdEndRenderPass(a_cmdBuff);
}

namespace vk_utils
{

QuadRenderer::~QuadRenderer()
{
  if (m_pipeline != nullptr)
  {
    vkDestroyPipeline(m_device, m_pipeline, nullptr);
    vkDestroyPipelineLayout(m_device, m_layout, nullptr);
    vkDestroyRenderPass(m_device, m_renderPass, nullptr);
  }

  if (m_fbTarget != nullptr)
    vkDestroyFramebuffer(m_device, m_fbTarget, NULL);

  if (m_dlayout != nullptr)
    vkDestroyDescriptorSetLayout(m_device, m_dlayout, NULL);
}

void QuadRenderer::Create(VkDevice a_device, const char* a_vspath, const char* a_fspath, vk_utils::RenderTargetInfo2D a_rtInfo)
{
  m_device = a_device;
  m_fbSize = a_rtInfo.size;
  m_rtCreateInfo = a_rtInfo;

  auto vertShaderCode = vk_utils::readSPVFile(a_vspath);
  auto fragShaderCode = vk_utils::readSPVFile(a_fspath);

  if (vertShaderCode.size() == 0 || fragShaderCode.size() == 0)
    RUN_TIME_ERROR("[FSQuad::Create]: can not load shaders");

  VkShaderModule vertShaderModule = vk_utils::createShaderModule(a_device, vertShaderCode);
  VkShaderModule fragShaderModule = vk_utils::createShaderModule(a_device, fragShaderCode);

  // create pipeline layout first
  //
  VkPipelineShaderStageCreateInfo vertShaderStageInfo = {};
  vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
  vertShaderStageInfo.module = vertShaderModule;
  vertShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo fragShaderStageInfo = {};
  fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  fragShaderStageInfo.module = fragShaderModule;
  fragShaderStageInfo.pName = "main";

  VkPipelineShaderStageCreateInfo      shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };
  VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
  vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputInfo.vertexBindingDescriptionCount = 0;
  vertexInputInfo.vertexAttributeDescriptionCount = 0;

  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // VkRect2D scissor; // #TODO: FIX THIS !!!, THIS IS DEBUG CODE
  // {
  //   scissor.offset = VkOffset2D{0,0};
  //   scissor.extent = VkExtent2D{512,512};
  // }
  // rect = scissor;
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  VkViewport viewport = {};
  viewport.x        = 0.0f;
  viewport.y        = 0.0f;
  viewport.width    = (float)rect.extent.width;
  viewport.height   = (float)rect.extent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  VkPipelineViewportStateCreateInfo viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports    = &viewport;
  viewportState.scissorCount  = 1;
  viewportState.pScissors     = &rect;

  VkPipelineRasterizationStateCreateInfo rasterizer = {};
  rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
  rasterizer.lineWidth = 1.0f;
  rasterizer.cullMode = VK_CULL_MODE_NONE; // VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable = VK_FALSE;

  VkPipelineMultisampleStateCreateInfo multisampling = {};
  multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;


  VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable = VK_FALSE;

  VkPipelineColorBlendStateCreateInfo colorBlending = {};
  colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable = VK_FALSE;
  colorBlending.logicOp = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount = 1;
  colorBlending.pAttachments = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  // create ds layout for binding texture shader 
  //
  {
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[1];

    descriptorSetLayoutBinding[0].binding = 0;
    descriptorSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    descriptorSetLayoutBinding[0].descriptorCount = 1;
    descriptorSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    descriptorSetLayoutBinding[0].pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 1;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBinding;

    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &m_dlayout));
  }

  VkPushConstantRange pcRange = {};
  pcRange.stageFlags = (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
  pcRange.offset = 0;
  pcRange.size = 8 * sizeof(float);

  VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
  pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.setLayoutCount = 0;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges = &pcRange;
  pipelineLayoutInfo.pSetLayouts = &m_dlayout;
  pipelineLayoutInfo.setLayoutCount = 1;

  VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, &m_layout));

  m_renderPass = vk_utils::createRenderPass(m_device, a_rtInfo);

  // finally create graphics pipeline
  //
  VkGraphicsPipelineCreateInfo pipelineInfo = {};
  pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.stageCount = 2;
  pipelineInfo.pStages = shaderStages;
  pipelineInfo.pVertexInputState = &vertexInputInfo;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState = &multisampling;
  pipelineInfo.pColorBlendState = &colorBlending;
  pipelineInfo.layout = m_layout;
  pipelineInfo.renderPass = m_renderPass;
  pipelineInfo.subpass = 0;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

  VK_CHECK_RESULT(vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

  vkDestroyShaderModule(m_device, fragShaderModule, nullptr);
  vkDestroyShaderModule(m_device, vertShaderModule, nullptr);
}

void QuadRenderer::SetRenderTarget(VkImageView a_imageView)
{
  if (m_fbTarget != nullptr)
    vkDestroyFramebuffer(m_device, m_fbTarget, NULL);

  //#TODO: add framebuffer cache inside this class ... 

  VkImageView attachments[] = { a_imageView };

  VkFramebufferCreateInfo framebufferInfo = {};
  framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  framebufferInfo.renderPass = m_renderPass;
  framebufferInfo.attachmentCount = 1;
  framebufferInfo.pAttachments = attachments;
  framebufferInfo.width = m_rtCreateInfo.size.width;
  framebufferInfo.height = m_rtCreateInfo.size.height;
  framebufferInfo.layers = 1;

  VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_fbTarget));

  m_targetView = a_imageView;
}

void QuadRenderer::DrawCmd(VkCommandBuffer a_cmdBuff, VkDescriptorSet a_inTexDescriptor, float a_offsAndScale[4])
{
  (void)a_offsAndScale;
  VkRenderPassBeginInfo renderPassInfo = {};
  renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderPassInfo.renderPass = m_renderPass;
  renderPassInfo.framebuffer = m_fbTarget;
  renderPassInfo.renderArea.offset = { 0, 0 };
  renderPassInfo.renderArea.extent = m_fbSize;

  VkClearValue clearValues[1] = {};
  clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
  renderPassInfo.clearValueCount = 1;
  renderPassInfo.pClearValues = &clearValues[0];
   
  //VkRect2D scissor{};
  //scissor.offset = {0, 0};
  //scissor.extent = {1024, 1024};
  //vkCmdSetScissor(a_cmdBuff, 0, 1, &scissor); // this should be enabled for pipeline, can't require that ("when the graphics pipeline is created with VK_DYNAMIC_STATE_SCISSOR")

  vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
   
  vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
  vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_layout, 0, 1, &a_inTexDescriptor, 0, NULL);

  vkCmdDraw(a_cmdBuff, 3, 1, 0, 0);

  vkCmdEndRenderPass(a_cmdBuff);
}



}
