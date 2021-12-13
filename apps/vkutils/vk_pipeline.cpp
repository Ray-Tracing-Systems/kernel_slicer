#include "vk_pipeline.h"
#include "vk_utils.h"

#ifdef __ANDROID__
namespace vk_android
{
  extern AAssetManager *g_pMgr;
}
#endif

VkPipelineInputAssemblyStateCreateInfo vk_utils::IA_TList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

VkPipelineInputAssemblyStateCreateInfo vk_utils::IA_PList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

VkPipelineInputAssemblyStateCreateInfo vk_utils::IA_LList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

VkPipelineInputAssemblyStateCreateInfo vk_utils::IA_LSList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

void vk_utils::GraphicsPipelineMaker::LoadShaders(VkDevice a_device, const std::unordered_map<VkShaderStageFlagBits, std::string> &shader_paths)
{
  int top = 0;
  for(auto& [stage, path] : shader_paths)
  {
    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage  = stage;

#ifdef __ANDROID__
    auto shaderCode = vk_utils::readSPVFile(vk_android::g_pMgr, path.c_str());
#else
    auto shaderCode = vk_utils::readSPVFile(path.c_str());
#endif
    VkShaderModule shaderModule = vk_utils::createShaderModule(a_device, shaderCode);
    shaderModules[top]          = shaderModule;

    stage_info.module = shaderModule;
    stage_info.pName  = "main";

    shaderStageInfos[top] = stage_info;
    top++;
  }

  m_stagesNum = top;
}

VkPipelineLayout vk_utils::GraphicsPipelineMaker::MakeLayout(VkDevice a_device, std::vector<VkDescriptorSetLayout> a_dslayouts,
  uint32_t a_pcRangeSize)
{
  auto m_device = a_device;

  pcRange.stageFlags = 0;
  for (unsigned i = 0; i < m_stagesNum; ++i)
    pcRange.stageFlags |= shaderStageInfos[i].stage;
  pcRange.offset     = 0;
  pcRange.size       = a_pcRangeSize;

  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pcRange;

  if(!a_dslayouts.empty())
  {
    pipelineLayoutInfo.pSetLayouts            = a_dslayouts.data();
    pipelineLayoutInfo.setLayoutCount         = a_dslayouts.size();
  }
  else
  {
    pipelineLayoutInfo.pSetLayouts            = VK_NULL_HANDLE;
    pipelineLayoutInfo.setLayoutCount         = 0;
  }

  VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

  return m_pipelineLayout;
}

void vk_utils::GraphicsPipelineMaker::SetDefaultState(uint32_t a_width, uint32_t a_height)
{
  VkExtent2D a_screenExtent{ a_width, a_height };

  viewport.x        = 0.0f;
  viewport.y        = 0.0f;
  viewport.width    = (float)a_screenExtent.width;
  viewport.height   = (float)a_screenExtent.height;
  viewport.minDepth = 0.0f;
  viewport.maxDepth = 1.0f;

  scissor.offset   = { 0, 0 };
  scissor.extent   = a_screenExtent;

  viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.viewportCount = 1;
  viewportState.pViewports    = &viewport;
  viewportState.scissorCount  = 1;
  viewportState.pScissors     = &scissor;

  rasterizer.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizer.depthClampEnable        = VK_FALSE;
  rasterizer.rasterizerDiscardEnable = VK_FALSE;
  rasterizer.polygonMode             = VK_POLYGON_MODE_FILL; // VK_POLYGON_MODE_FILL; // VK_POLYGON_MODE_LINE
  rasterizer.lineWidth               = 1.0f;
  rasterizer.cullMode                = VK_CULL_MODE_NONE; // VK_CULL_MODE_BACK_BIT;
  rasterizer.frontFace               = VK_FRONT_FACE_CLOCKWISE;
  rasterizer.depthBiasEnable         = VK_FALSE;

  multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampling.sampleShadingEnable  = VK_FALSE;
  multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  colorBlendAttachment.blendEnable    = VK_FALSE;

  colorBlending.sType             = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlending.logicOpEnable     = VK_FALSE;
  colorBlending.logicOp           = VK_LOGIC_OP_CLEAR;
  colorBlending.attachmentCount   = 1;
  colorBlending.pAttachments      = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  depthStencilTest.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencilTest.depthTestEnable       = true;
  depthStencilTest.depthWriteEnable      = true;
  depthStencilTest.depthCompareOp        = VK_COMPARE_OP_LESS_OR_EQUAL;
  depthStencilTest.depthBoundsTestEnable = false;
  depthStencilTest.stencilTestEnable     = false;
  depthStencilTest.depthBoundsTestEnable = VK_FALSE;
  depthStencilTest.minDepthBounds        = 0.0f; // Optional
  depthStencilTest.maxDepthBounds        = 1.0f; // Optional
}

VkPipeline vk_utils::GraphicsPipelineMaker::MakePipeline(VkDevice a_device, VkPipelineVertexInputStateCreateInfo a_vertexLayout,
                                                         VkRenderPass a_renderPass,
                                                         std::vector<VkDynamicState> a_dynamicStates,
                                                         VkPipelineInputAssemblyStateCreateInfo a_inputAssembly)
{
  inputAssembly = a_inputAssembly;

  VkPipelineDynamicStateCreateInfo dynamicState = {};
  dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = a_dynamicStates.size();
  dynamicState.pDynamicStates    = a_dynamicStates.data();

  pipelineInfo = {};
  pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  pipelineInfo.flags               = 0;
  pipelineInfo.stageCount          = m_stagesNum;
  pipelineInfo.pStages             = shaderStageInfos;
  pipelineInfo.pVertexInputState   = &a_vertexLayout;
  pipelineInfo.pInputAssemblyState = &inputAssembly;
  pipelineInfo.pViewportState      = &viewportState;
  pipelineInfo.pRasterizationState = &rasterizer;
  pipelineInfo.pMultisampleState   = &multisampling;
  pipelineInfo.pColorBlendState    = &colorBlending;
  pipelineInfo.layout              = m_pipelineLayout;
  pipelineInfo.renderPass          = a_renderPass;
  pipelineInfo.subpass             = 0;
  pipelineInfo.pDynamicState       = &dynamicState;
  pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;
  pipelineInfo.pDepthStencilState  = &depthStencilTest;

  VK_CHECK_RESULT(vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

  for (size_t i = 0; i < m_stagesNum; ++i)
  {
    if(shaderModules[i] != VK_NULL_HANDLE)
      vkDestroyShaderModule(a_device, shaderModules[i], VK_NULL_HANDLE);
    shaderModules[i] = VK_NULL_HANDLE;
  }

  return m_pipeline;
}


////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////


void vk_utils::ComputePipelineMaker::LoadShader(VkDevice a_device, const std::string& a_shaderPath,
                                                const VkSpecializationInfo *a_specInfo, const char* a_mainName)
{
  m_mainName = a_mainName;

  shaderStageInfo = {};
  shaderStageInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  shaderStageInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;

#ifdef __ANDROID__
  auto shaderCode = vk_utils::readSPVFile(vk_android::g_pMgr, a_shaderPath.c_str());
#else
  auto shaderCode = vk_utils::readSPVFile(a_shaderPath.c_str());
#endif
  shaderModule    = vk_utils::createShaderModule(a_device, shaderCode);

  shaderStageInfo.module = shaderModule;
  shaderStageInfo.pName  = m_mainName.c_str();
}

VkPipelineLayout vk_utils::ComputePipelineMaker::MakeLayout(VkDevice a_device, std::vector<VkDescriptorSetLayout> a_dslayouts,
                                                            uint32_t a_pcRangeSize)
{

  assert(a_device);

  if (a_pcRangeSize)
  {
    pcRange.stageFlags = shaderStageInfo.stage;
    pcRange.offset     = 0;
    pcRange.size       = a_pcRangeSize;
  }

  pipelineLayoutInfo                        = {};
  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = a_pcRangeSize ? 1 : 0;
  pipelineLayoutInfo.pPushConstantRanges    = a_pcRangeSize ? &pcRange : nullptr;

  if (!a_dslayouts.empty())
  {
    pipelineLayoutInfo.pSetLayouts    = a_dslayouts.data();
    pipelineLayoutInfo.setLayoutCount = a_dslayouts.size();
  }
  else
  {
    pipelineLayoutInfo.pSetLayouts    = VK_NULL_HANDLE;
    pipelineLayoutInfo.setLayoutCount = 0;
  }

  VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

  return m_pipelineLayout;
}

VkPipeline vk_utils::ComputePipelineMaker::MakePipeline(VkDevice a_device)
{
  pipelineInfo                    = {};
  pipelineInfo.sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  pipelineInfo.flags              = 0;
  pipelineInfo.stage              = shaderStageInfo;
  pipelineInfo.layout             = m_pipelineLayout;
  pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;
  VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline));

  if (shaderModule != VK_NULL_HANDLE)
    vkDestroyShaderModule(a_device, shaderModule, VK_NULL_HANDLE);

  return m_pipeline;
}
