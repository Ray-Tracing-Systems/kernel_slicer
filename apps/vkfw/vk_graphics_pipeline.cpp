#include "vk_graphics_pipeline.h"

#include <cstring>

vkfw::GraphicsPipelineMaker::GraphicsPipelineMaker()
{ 
  memset(this, 0, sizeof(GraphicsPipelineMaker)); ///<! ASSUME THIS IS PLAIN OLD DATA!!!!
}

#if defined(__ANDROID__)
void vkfw::GraphicsPipelineMaker::CreateShaders(AAssetManager* mgr, VkDevice a_device, const std::unordered_map<VkShaderStageFlagBits, std::string>& shader_paths)
{
  int top = 0;
  for (auto& shader : shader_paths)
  {
    auto stage = shader.first;
    auto path = shader.second;

    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = stage;


    // Read the file
    assert(mgr);
    AAsset* file = AAssetManager_open(mgr, path.c_str(), AASSET_MODE_BUFFER);
    size_t fileLength = AAsset_getLength(file);
    char* fileContent = new char[fileLength];

    AAsset_read(file, fileContent, fileLength);
    AAsset_close(file);

    VkShaderModuleCreateInfo shaderModuleCreateInfo{
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .codeSize = fileLength,
        .pCode = (const uint32_t*)fileContent,
        .flags = 0,
    };

    VkShaderModule shaderModule;
    VkResult result = vkCreateShaderModule(a_device, &shaderModuleCreateInfo, nullptr, &shaderModule);
    assert(result == VK_SUCCESS);
    delete[] fileContent;

  
    shaderModules[top] = shaderModule;
    stage_info.module = shaderModule;
    stage_info.pName = "main";
    shaderStageInfos[top] = stage_info;
    top++;
  }

  m_stagesNum = top;
}
#else
void vkfw::GraphicsPipelineMaker::CreateShaders(VkDevice a_device, const std::unordered_map<VkShaderStageFlagBits, std::string> &shader_paths)
{
  int top = 0;
  for(auto& shader : shader_paths)
  {
    auto stage = shader.first;
    auto path  = shader.second;

    VkPipelineShaderStageCreateInfo stage_info = {};
    stage_info.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage  = stage;

    auto shaderCode             = vk_utils::ReadFile(path.c_str());
    VkShaderModule shaderModule = vk_utils::CreateShaderModule(a_device, shaderCode);
    shaderModules[top]          = shaderModule;
    
    stage_info.module = shaderModule;
    stage_info.pName  = "main";
    
    shaderStageInfos[top] = stage_info;
    top++;
  }

  m_stagesNum = top;
}
#endif

VkPipelineInputAssemblyStateCreateInfo vkfw::IA_TList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

VkPipelineInputAssemblyStateCreateInfo vkfw::IA_PList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

VkPipelineInputAssemblyStateCreateInfo vkfw::IA_LList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}

VkPipelineInputAssemblyStateCreateInfo vkfw::IA_LSList()
{
  VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
  inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
  inputAssembly.primitiveRestartEnable = VK_FALSE;
  return inputAssembly;
}


VkPipelineLayout vkfw::GraphicsPipelineMaker::MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout, uint32_t a_pcRangeSize)
{
  auto m_device = a_device;

  pcRange.stageFlags = 0;
  for (int i = 0; i < m_stagesNum; ++i)
    pcRange.stageFlags |= shaderStageInfos[i].stage;
  pcRange.offset     = 0;
  pcRange.size       = a_pcRangeSize; 

  pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipelineLayoutInfo.pushConstantRangeCount = 0;
  pipelineLayoutInfo.pushConstantRangeCount = 1;
  pipelineLayoutInfo.pPushConstantRanges    = &pcRange;

  if(a_dslayout != VK_NULL_HANDLE)
  {
    pipelineLayoutInfo.pSetLayouts            = &a_dslayout; //&descriptorSetLayout;
    pipelineLayoutInfo.setLayoutCount         = 1;
  }
  else
  {
    pipelineLayoutInfo.pSetLayouts            = VK_NULL_HANDLE;
    pipelineLayoutInfo.setLayoutCount         = 0;
  }

  VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout));

  return m_pipelineLayout;
}

void vkfw::GraphicsPipelineMaker::SetDefault_Simple3D(uint32_t a_width, uint32_t a_height)
{
  VkExtent2D a_screenExtent{ uint32_t(a_width), uint32_t(a_height) };

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
  colorBlending.logicOp           = VK_LOGIC_OP_COPY;
  colorBlending.attachmentCount   = 1;
  colorBlending.pAttachments      = &colorBlendAttachment;
  colorBlending.blendConstants[0] = 0.0f;
  colorBlending.blendConstants[1] = 0.0f;
  colorBlending.blendConstants[2] = 0.0f;
  colorBlending.blendConstants[3] = 0.0f;

  depthStencilTest.sType                 = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencilTest.depthTestEnable       = true;
  depthStencilTest.depthWriteEnable      = true;
  depthStencilTest.depthCompareOp        = VK_COMPARE_OP_LESS;
  depthStencilTest.depthBoundsTestEnable = false;
  depthStencilTest.stencilTestEnable     = false;
  depthStencilTest.depthBoundsTestEnable = VK_FALSE;
  depthStencilTest.minDepthBounds        = 0.0f; // Optional
  depthStencilTest.maxDepthBounds        = 1.0f; // Optional
}

VkPipeline vkfw::GraphicsPipelineMaker::MakePipeline(VkDevice a_device, VkPipelineVertexInputStateCreateInfo a_vertexLayout,
                                                     VkRenderPass a_renderPass, VkPipelineInputAssemblyStateCreateInfo a_inputAssembly)
{
  auto m_device = a_device;
  inputAssembly = a_inputAssembly;

  std::vector<VkDynamicState> dynamicStates = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };

  VkPipelineDynamicStateCreateInfo dynamicState = {};
  dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
  dynamicState.pDynamicStates    = dynamicStates.data();

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

  // free all shader modules due to we don't need them anymore
  //
  for (auto& module : shaderModules)
  {
    if(module != VK_NULL_HANDLE)
      vkDestroyShaderModule(a_device, module, VK_NULL_HANDLE);
    module = VK_NULL_HANDLE;
  }

  return m_pipeline;
}
