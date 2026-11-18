
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Graphics pipeline (LiteRS). Render targets always stay in VK_IMAGE_LAYOUT_GENERAL, so kernels and draws use them without
// layout transitions; LiteRS::Clear is a separate command, {% if GPDynamicRendering %}draws load and store attachments.
// Vulkan 1.3: dynamic rendering (no render passes and framebuffers) and extended dynamic state: 'cullMode', 'frontFace',
// 'depthTest', 'depthWrite' and 'depthCompare' of LiteRS::Pipeline are read at every draw, 'blendMode' is baked into pipelines.{% else %}render passes load and store attachments.
// Vulkan 1.0: render pass and framebuffer for each LiteRS::Pipeline member, its state is baked into pipelines.{% endif %}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

{% if GPDynamicRendering %}
static VkRenderingAttachmentInfo GPMakeRenderingAttachment(VkImageView a_view)
{
  VkRenderingAttachmentInfo info = {};
  info.sType       = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
  info.imageView   = a_view;
  info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
  info.loadOp      = VK_ATTACHMENT_LOAD_OP_LOAD;
  info.storeOp     = VK_ATTACHMENT_STORE_OP_STORE;
  return info;
}
{% else %}
static VkAttachmentDescription GPMakeAttachment(VkFormat a_format)
{
  VkAttachmentDescription desc = {};
  desc.format         = a_format;
  desc.samples        = VK_SAMPLE_COUNT_1_BIT;
  desc.loadOp         = VK_ATTACHMENT_LOAD_OP_LOAD;
  desc.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
  desc.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  desc.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  desc.initialLayout  = VK_IMAGE_LAYOUT_GENERAL;
  desc.finalLayout    = VK_IMAGE_LAYOUT_GENERAL;
  return desc;
}
{% endif %}

static VkPipelineColorBlendAttachmentState GPMakeBlendState(LiteRS::BlendMode a_mode, bool a_integerTarget)
{
  VkPipelineColorBlendAttachmentState state = {};
  state.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
  state.blendEnable    = (a_mode != LiteRS::BlendMode::NONE && !a_integerTarget) ? VK_TRUE : VK_FALSE; // integer (RGBA8 packed to uint) targets can't be blended
  state.colorBlendOp   = VK_BLEND_OP_ADD;
  state.alphaBlendOp   = VK_BLEND_OP_ADD;
  if(a_mode == LiteRS::BlendMode::ALPHA)
  {
    state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  }
  else
  {
    state.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    state.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
    state.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    state.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  }
  return state;
}

// Render targets have their own allocations so a growing viewport does not re-upload
// geometry, textures, random generators or accumulation buffers.
void {{MainClassName}}{{MainClassSuffix}}::InitGraphicsTargets()
{
  {% for Target in GPRenderTargets %}
  {
    auto& image = m_vdata.{{Target.Name}}Texture;
    image = CreateTexture2D({{Target.Name}}{{Target.AccessSymb}}width(), {{Target.Name}}{{Target.AccessSymb}}height(), VkFormat({{Target.Format}}), {{Target.Usage}});
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(m_device, image, &req);
    VkMemoryAllocateInfo alloc = {};
    alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = vk_utils::findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(m_device, &alloc, nullptr, &m_gpMemory_{{Target.Name}}));
    VK_CHECK_RESULT(vkBindImageMemory(m_device, image, m_gpMemory_{{Target.Name}}, 0));
    m_vdata.{{Target.Name}}View = CreateView(VkFormat({{Target.Format}}), image);
    m_gpSize_{{Target.Name}} = { {{Target.Name}}{{Target.AccessSymb}}width(), {{Target.Name}}{{Target.AccessSymb}}height() };
  }
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::DestroyGraphicsTargets()
{
  {% for Target in GPRenderTargets %}
  vkDestroyImageView(m_device, m_vdata.{{Target.Name}}View, nullptr);
  vkDestroyImage(m_device, m_vdata.{{Target.Name}}Texture, nullptr);
  vkFreeMemory(m_device, m_gpMemory_{{Target.Name}}, nullptr);
  m_vdata.{{Target.Name}}View = VK_NULL_HANDLE;
  m_vdata.{{Target.Name}}Texture = VK_NULL_HANDLE;
  m_gpMemory_{{Target.Name}} = VK_NULL_HANDLE;
  m_gpSize_{{Target.Name}} = {};
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::ResizeGraphicsTargets()
{
  if(m_commitCount == 0)
    return;
  bool changed = false;
  {% for Target in GPRenderTargets %}
  changed = changed || m_gpSize_{{Target.Name}}.width != {{Target.Name}}{{Target.AccessSymb}}width() || m_gpSize_{{Target.Name}}.height != {{Target.Name}}{{Target.AccessSymb}}height();
  {% endfor %}
  if(!changed)
    return;
  VK_CHECK_RESULT(vkDeviceWaitIdle(m_device));
  DestroyGraphics();
  DestroyGraphicsTargets();
  InitGraphicsTargets();

  auto copyHelper = m_pLastCopyHelper;
  VkCommandBuffer cmd = copyHelper->CmdBuffer();
  VK_CHECK_RESULT(vkResetCommandBuffer(cmd, 0));
  VkCommandBufferBeginInfo begin = {};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  VK_CHECK_RESULT(vkBeginCommandBuffer(cmd, &begin));
  {% for Target in GPRenderTargets %}
  {
    VkImageMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.image = m_vdata.{{Target.Name}}Texture;
    barrier.subresourceRange = { {{Target.Aspect}}, 0, 1, 0, 1 };
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
  }
  {% endfor %}
  VK_CHECK_RESULT(vkEndCommandBuffer(cmd));
  vk_utils::executeCommandBufferNow(cmd, copyHelper->TransferQueue(), m_device);

  {% for MainFunc in MainFunctions %}
  {% for DS in MainFunc.DescriptorSets %}
  {% for Arg in DS.Args %}
  {% if Arg.IsTexture %}
  {% for Target in GPRenderTargets %}
  {% if Arg.NameOriginal == Target.Name %}
  {
    VkDescriptorImageInfo image = { {{Arg.SamplerName}}, m_vdata.{{Target.Name}}View, VK_IMAGE_LAYOUT_GENERAL };
    VkWriteDescriptorSet write = {};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_allGeneratedDS[{{DS.Id}}];
    write.dstBinding = {{Arg.Id}};
    write.descriptorCount = 1;
    write.descriptorType = {{Arg.AccessDSType}};
    write.pImageInfo = &image;
    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
  }
  {% endif %}
  {% endfor %}
  {% endif %}
  {% endfor %}
  {% endfor %}
  {% endfor %}
  InitGraphics();
}

void {{MainClassName}}{{MainClassSuffix}}::DestroyGraphics()
{
  {% if HasGPIndirect %}
  if(m_gpIndirectPipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, m_gpIndirectPipeline, nullptr);
  if(m_gpIndirectLayout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(m_device, m_gpIndirectLayout, nullptr);
  if(m_gpIndirectDSPool != VK_NULL_HANDLE)
    vkDestroyDescriptorPool(m_device, m_gpIndirectDSPool, nullptr); // frees m_gpIndirectDS
  if(m_gpIndirectDSLayout != VK_NULL_HANDLE)
    vkDestroyDescriptorSetLayout(m_device, m_gpIndirectDSLayout, nullptr);
  m_gpIndirectPipeline = VK_NULL_HANDLE;
  m_gpIndirectLayout   = VK_NULL_HANDLE;
  m_gpIndirectDSPool   = VK_NULL_HANDLE;
  m_gpIndirectDS       = VK_NULL_HANDLE;
  m_gpIndirectDSLayout = VK_NULL_HANDLE;
  {% endif %}
  {% for Draw in GPDraws %}
  if({{Draw.Name}}Pipeline != VK_NULL_HANDLE)
    vkDestroyPipeline(m_device, {{Draw.Name}}Pipeline, nullptr);
  if({{Draw.Name}}Layout != VK_NULL_HANDLE)
    vkDestroyPipelineLayout(m_device, {{Draw.Name}}Layout, nullptr);
  {{Draw.Name}}Pipeline = VK_NULL_HANDLE;
  {{Draw.Name}}Layout   = VK_NULL_HANDLE;
  {% endfor %}
  {% for Pipe in GPPipelines %}
  if(m_gp_{{Pipe.Name}}.framebuffer != VK_NULL_HANDLE)
    vkDestroyFramebuffer(m_device, m_gp_{{Pipe.Name}}.framebuffer, nullptr);
  if(m_gp_{{Pipe.Name}}.renderPass != VK_NULL_HANDLE)
    vkDestroyRenderPass(m_device, m_gp_{{Pipe.Name}}.renderPass, nullptr);
  m_gp_{{Pipe.Name}} = GPPassObjects();
  {% endfor %}
}

void {{MainClassName}}{{MainClassSuffix}}::InitGraphics()
{
  DestroyGraphics();
  {% if HasGPIndirect %}

  // compute pipeline which writes counts of indirect draws: set 0 = { class data (UBO), m_gpIndirectBuffer }
  {
    VkDescriptorSetLayoutBinding bindings[2] = {};
    for(uint32_t i = 0; i < 2; i++)
    {
      bindings[i].binding         = i;
      bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      bindings[i].descriptorCount = 1;
      bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dsLayoutInfo = {};
    dsLayoutInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dsLayoutInfo.bindingCount = 2;
    dsLayoutInfo.pBindings    = bindings;
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(m_device, &dsLayoutInfo, nullptr, &m_gpIndirectDSLayout));

    const VkDescriptorPoolSize poolSize = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo poolInfo = {};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets       = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    VK_CHECK_RESULT(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_gpIndirectDSPool));

    VkDescriptorSetAllocateInfo allocInfo = {};
    allocInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool     = m_gpIndirectDSPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts        = &m_gpIndirectDSLayout;
    VK_CHECK_RESULT(vkAllocateDescriptorSets(m_device, &allocInfo, &m_gpIndirectDS));

    const VkDescriptorBufferInfo bufferInfos[2] = { {m_classDataBuffer, 0, VK_WHOLE_SIZE}, {m_gpIndirectBuffer, 0, VK_WHOLE_SIZE} };
    VkWriteDescriptorSet writes[2] = {};
    for(uint32_t i = 0; i < 2; i++)
    {
      writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet          = m_gpIndirectDS;
      writes[i].dstBinding      = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      writes[i].pBufferInfo     = &bufferInfos[i];
    }
    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);

    const VkPushConstantRange  pcRange    = { VK_SHADER_STAGE_COMPUTE_BIT, 0, 3*sizeof(uint32_t) }; // draw id, counts computed on host
    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = 1;
    layoutInfo.pSetLayouts            = &m_gpIndirectDSLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_gpIndirectLayout));

    const std::string shaderPath = AlterShaderPath("{{ShaderFolder}}/z_gp_update_indirect.comp.spv");
    auto           code          = vk_utils::readSPVFile(shaderPath.c_str());
    VkShaderModule module        = vk_utils::createShaderModule(m_device, code);

    VkComputePipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType        = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = module;
    pipelineInfo.stage.pName  = "main";
    pipelineInfo.layout       = m_gpIndirectLayout;
    VK_CHECK_RESULT(vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_gpIndirectPipeline));
    vkDestroyShaderModule(m_device, module, nullptr);
  }
  {% endif %}
  {% for Pipe in GPPipelines %}

  {% if GPDynamicRendering %}
  m_gp_{{Pipe.Name}}.width  = {{Pipe.SizeFrom}}.width();  // dynamic rendering: attachments are set at every draw
  m_gp_{{Pipe.Name}}.height = {{Pipe.SizeFrom}}.height();
  {% else %}
  // render pass and framebuffer for '{{Pipe.Name}}'
  {
    std::vector<VkAttachmentDescription> attachments;
    std::vector<VkAttachmentReference>   colorRefs;
    std::vector<VkImageView>             views;
    {% for Target in Pipe.ColorTargets %}
    attachments.push_back(GPMakeAttachment({{Target.Format}}));
    colorRefs.push_back(VkAttachmentReference{uint32_t(attachments.size() - 1), VK_IMAGE_LAYOUT_GENERAL});
    views.push_back(m_vdata.{{Target.Name}}View);
    {% endfor %}
    {% if Pipe.HasDepth %}
    attachments.push_back(GPMakeAttachment(VK_FORMAT_D32_SFLOAT));
    VkAttachmentReference depthRef = {uint32_t(attachments.size() - 1), VK_IMAGE_LAYOUT_GENERAL};
    views.push_back(m_vdata.{{Pipe.DepthTarget}}View);
    {% endif %}

    VkSubpassDescription subpass    = {};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = uint32_t(colorRefs.size());
    subpass.pColorAttachments       = colorRefs.data();
    subpass.pDepthStencilAttachment = {% if Pipe.HasDepth %}&depthRef{% else %}nullptr{% endif %};

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = uint32_t(attachments.size());
    renderPassInfo.pAttachments    = attachments.data();
    renderPassInfo.subpassCount    = 1;
    renderPassInfo.pSubpasses      = &subpass;
    VK_CHECK_RESULT(vkCreateRenderPass(m_device, &renderPassInfo, nullptr, &m_gp_{{Pipe.Name}}.renderPass));

    m_gp_{{Pipe.Name}}.width  = {{Pipe.SizeFrom}}.width();
    m_gp_{{Pipe.Name}}.height = {{Pipe.SizeFrom}}.height();

    VkFramebufferCreateInfo framebufferInfo = {};
    framebufferInfo.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    framebufferInfo.renderPass      = m_gp_{{Pipe.Name}}.renderPass;
    framebufferInfo.attachmentCount = uint32_t(views.size());
    framebufferInfo.pAttachments    = views.data();
    framebufferInfo.width           = m_gp_{{Pipe.Name}}.width;
    framebufferInfo.height          = m_gp_{{Pipe.Name}}.height;
    framebufferInfo.layers          = 1;
    VK_CHECK_RESULT(vkCreateFramebuffer(m_device, &framebufferInfo, nullptr, &m_gp_{{Pipe.Name}}.framebuffer));
  }
  {% endif %}
  {% endfor %}
  {% for Draw in GPDraws %}

  {% if GPDynamicRendering %}
  // graphics pipeline '{{Draw.Name}}': blend mode is taken from '{{Draw.Pipeline}}', call CommitDeviceData() again if you change it
  {% else %}
  // graphics pipeline '{{Draw.Name}}': state is taken from '{{Draw.Pipeline}}', call CommitDeviceData() again if you change it
  {% endif %}
  {
    const LiteRS::Pipeline& state = {{Draw.Pipeline}};

    VkDescriptorSetLayout setLayouts[2] = { {{Draw.VS}}DSLayout{% if Draw.HasPS %}, {{Draw.PS}}DSLayout{% endif %} };
    VkPushConstantRange   pcRange       = { VK_SHADER_STAGE_VERTEX_BIT, 0, uint32_t(sizeof({{Draw.Name}}_PC)) };

    VkPipelineLayoutCreateInfo layoutInfo = {};
    layoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount         = {% if Draw.HasPS %}2{% else %}1{% endif %};
    layoutInfo.pSetLayouts            = setLayouts;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges    = &pcRange;
    VK_CHECK_RESULT(vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &{{Draw.Name}}Layout));

    std::vector<VkPipelineShaderStageCreateInfo> stages;
    std::vector<VkShaderModule>                  modules;
    {
      const std::string vsPath = AlterShaderPath("{{ShaderFolder}}/{{Draw.VSFile}}");
      auto vsCode = vk_utils::readSPVFile(vsPath.c_str());
      modules.push_back(vk_utils::createShaderModule(m_device, vsCode));
      VkPipelineShaderStageCreateInfo stage = {};
      stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
      stage.module = modules.back();
      stage.pName  = "main";
      stages.push_back(stage);
    }
    {% if Draw.HasPS %}
    {
      const std::string psPath = AlterShaderPath("{{ShaderFolder}}/{{Draw.PSFile}}");
      auto psCode = vk_utils::readSPVFile(psPath.c_str());
      modules.push_back(vk_utils::createShaderModule(m_device, psCode));
      VkPipelineShaderStageCreateInfo stage = {};
      stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      stage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
      stage.module = modules.back();
      stage.pName  = "main";
      stages.push_back(stage);
    }
    {% endif %}

    VkPipelineVertexInputStateCreateInfo vertexInput = {}; // vertex shader reads vertex data from buffers itself
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType    = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState = {}; // viewport and scissor are dynamic
    viewportState.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount  = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer = {};
    rasterizer.sType       = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode    = VkCullModeFlags(uint32_t(state.cullMode));  // LiteRS enums have the same values as Vulkan ones
    rasterizer.frontFace   = VkFrontFace(uint32_t(state.frontFace));
    rasterizer.lineWidth   = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling = {};
    multisampling.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil = {};
    depthStencil.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable  = ({{Draw.HasDepth}} && state.depthTest) ? VK_TRUE : VK_FALSE;
    depthStencil.depthWriteEnable = ({{Draw.HasDepth}} && state.depthTest && state.depthWrite) ? VK_TRUE : VK_FALSE;
    depthStencil.depthCompareOp   = VkCompareOp(uint32_t(state.depthCompare));

    std::vector<VkPipelineColorBlendAttachmentState> blendStates;
    {% if GPDynamicRendering %}
    std::vector<VkFormat> colorFormats;
    {% for Target in Draw.PassTargets %}
    blendStates.push_back(GPMakeBlendState(state.blendMode, {{Target.IsUint}}));
    {% if not Draw.HasPS %}
    blendStates.back().colorWriteMask = 0; // pass without pixel shader writes only depth
    {% endif %}
    colorFormats.push_back({{Target.Format}});
    {% endfor %}
    {% else %}
    {% for Target in Draw.ColorTargets %}
    blendStates.push_back(GPMakeBlendState(state.blendMode, {{Target.IsUint}}));
    {% endfor %}
    {% endif %}
    VkPipelineColorBlendStateCreateInfo colorBlending = {};
    colorBlending.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.attachmentCount = uint32_t(blendStates.size());
    colorBlending.pAttachments    = blendStates.data();

    {% if GPDynamicRendering %}
    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
                                       VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
                                       VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP };
    {% else %}
    VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
    {% endif %}
    VkPipelineDynamicStateCreateInfo dynamicState = {};
    dynamicState.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = uint32_t(sizeof(dynamicStates)/sizeof(dynamicStates[0]));
    dynamicState.pDynamicStates    = dynamicStates;
    {% if GPDynamicRendering %}

    VkPipelineRenderingCreateInfo renderingInfo = {};
    renderingInfo.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount    = uint32_t(colorFormats.size());
    renderingInfo.pColorAttachmentFormats = colorFormats.data();
    renderingInfo.depthAttachmentFormat   = {% if Draw.HasDepth %}VK_FORMAT_D32_SFLOAT{% else %}VK_FORMAT_UNDEFINED{% endif %};
    {% endif %}

    VkGraphicsPipelineCreateInfo pipelineInfo = {};
    pipelineInfo.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount          = uint32_t(stages.size());
    pipelineInfo.pStages             = stages.data();
    pipelineInfo.pVertexInputState   = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState      = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState   = &multisampling;
    pipelineInfo.pDepthStencilState  = &depthStencil;
    pipelineInfo.pColorBlendState    = &colorBlending;
    pipelineInfo.pDynamicState       = &dynamicState;
    pipelineInfo.layout              = {{Draw.Name}}Layout;
    {% if GPDynamicRendering %}
    pipelineInfo.pNext               = &renderingInfo;
    pipelineInfo.renderPass          = VK_NULL_HANDLE;
    {% else %}
    pipelineInfo.renderPass          = m_gp_{{Draw.Pipeline}}.renderPass;
    {% endif %}
    pipelineInfo.subpass             = 0;
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(m_device, m_pipelineCache, 1, &pipelineInfo, nullptr, &{{Draw.Name}}Pipeline));

    for(auto module : modules)
      vkDestroyShaderModule(m_device, module, nullptr);
  }
  {% endfor %}
}
{% for Pipe in GPPipelines %}

void {{MainClassName}}{{MainClassSuffix}}::Clear_{{Pipe.Name}}Cmd(VkCommandBuffer a_commandBuffer)
{
  const LiteRS::Pipeline& state = {{Pipe.Name}};
  {% if length(Pipe.ColorTargets) > 0 %}
  VkImageSubresourceRange colorRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
  {% endif %}
  {% for Target in Pipe.ColorTargets %}
  {
    VkClearColorValue clearValue = {};
    {% if Target.IsRawUint4 %}
    clearValue.uint32[0] = state.clearUintColor.x;
    clearValue.uint32[1] = state.clearUintColor.y;
    clearValue.uint32[2] = state.clearUintColor.z;
    clearValue.uint32[3] = state.clearUintColor.w;
    {% else if Target.IsUint %}
    clearValue.uint32[0] = LiteRS::detail::PackRGBA8(state.clearColor); // RGBA8 target is stored as uint
    {% else %}
    clearValue.float32[0] = state.clearColor.x;
    clearValue.float32[1] = state.clearColor.y;
    clearValue.float32[2] = state.clearColor.z;
    clearValue.float32[3] = state.clearColor.w;
    {% endif %}
    vkCmdClearColorImage(a_commandBuffer, m_vdata.{{Target.Name}}Texture, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &colorRange);
  }
  {% endfor %}
  {% if Pipe.HasDepth %}
  {
    VkImageSubresourceRange depthRange = { VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1 };
    VkClearDepthStencilValue clearValue = { state.clearDepth, 0 };
    vkCmdClearDepthStencilImage(a_commandBuffer, m_vdata.{{Pipe.DepthTarget}}Texture, VK_IMAGE_LAYOUT_GENERAL, &clearValue, 1, &depthRange);
  }
  {% endif %}
}
{% endfor %}
{% if HasGPIndirect %}

void {{MainClassName}}{{MainClassSuffix}}::GPUpdateIndirectCmd(VkCommandBuffer a_commandBuffer, uint32_t a_indirectId, uint32_t a_hostCount, uint32_t a_hostInstNum)
{
  const uint32_t args[3] = { a_indirectId, a_hostCount, a_hostInstNum };
  vkCmdBindPipeline      (a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_gpIndirectPipeline);
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_gpIndirectLayout, 0, 1, &m_gpIndirectDS, 0, nullptr);
  vkCmdPushConstants     (a_commandBuffer, m_gpIndirectLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, uint32_t(sizeof(args)), args);
  vkCmdDispatch          (a_commandBuffer, 1, 1, 1);
}
{% endif %}
{% for Draw in GPDraws %}

void {{MainClassName}}{{MainClassSuffix}}::{{Draw.Name}}Cmd(VkCommandBuffer a_commandBuffer, VkDescriptorSet a_dsVS, VkDescriptorSet a_dsPS, {% if Draw.Indexed %}VkBuffer a_indexBuffer, {% endif %}uint32_t a_count{% if Draw.Instanced %}, uint32_t a_instNum{% endif %}, uint32_t a_indirectId{% for Arg in Draw.UserArgs %}, {{Arg.Type}} {{Arg.Name}}{% endfor %})
{
  const LiteRS::Pipeline& state = {{Draw.Pipeline}};

  {% if GPDynamicRendering %}
  std::vector<VkRenderingAttachmentInfo> colorAttachments;
  {% for Target in Draw.PassTargets %}
  colorAttachments.push_back(GPMakeRenderingAttachment(m_vdata.{{Target.Name}}View));
  {% endfor %}
  {% if Draw.HasDepth %}
  const VkRenderingAttachmentInfo depthAttachment = GPMakeRenderingAttachment(m_vdata.{{Draw.DepthTarget}}View);
  {% endif %}

  VkRenderingInfo renderingInfo = {};
  renderingInfo.sType                    = VK_STRUCTURE_TYPE_RENDERING_INFO;
  renderingInfo.renderArea.extent.width  = m_gp_{{Draw.Pipeline}}.width;
  renderingInfo.renderArea.extent.height = m_gp_{{Draw.Pipeline}}.height;
  renderingInfo.layerCount               = 1;
  renderingInfo.colorAttachmentCount     = uint32_t(colorAttachments.size());
  renderingInfo.pColorAttachments        = colorAttachments.data();
  renderingInfo.pDepthAttachment         = {% if Draw.HasDepth %}&depthAttachment{% else %}nullptr{% endif %};
  vkCmdBeginRendering(a_commandBuffer, &renderingInfo);
  vkCmdBindPipeline(a_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, {{Draw.Name}}Pipeline);

  vkCmdSetCullMode       (a_commandBuffer, VkCullModeFlags(uint32_t(state.cullMode))); // LiteRS enums have the same values as Vulkan ones
  vkCmdSetFrontFace      (a_commandBuffer, VkFrontFace(uint32_t(state.frontFace)));
  vkCmdSetDepthTestEnable (a_commandBuffer, ({{Draw.HasDepth}} && state.depthTest) ? VK_TRUE : VK_FALSE);
  vkCmdSetDepthWriteEnable(a_commandBuffer, ({{Draw.HasDepth}} && state.depthTest && state.depthWrite) ? VK_TRUE : VK_FALSE);
  vkCmdSetDepthCompareOp (a_commandBuffer, VkCompareOp(uint32_t(state.depthCompare)));
  {% else %}
  VkRenderPassBeginInfo beginInfo = {};
  beginInfo.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  beginInfo.renderPass               = m_gp_{{Draw.Pipeline}}.renderPass;
  beginInfo.framebuffer              = m_gp_{{Draw.Pipeline}}.framebuffer;
  beginInfo.renderArea.extent.width  = m_gp_{{Draw.Pipeline}}.width;
  beginInfo.renderArea.extent.height = m_gp_{{Draw.Pipeline}}.height;
  vkCmdBeginRenderPass(a_commandBuffer, &beginInfo, VK_SUBPASS_CONTENTS_INLINE);
  vkCmdBindPipeline(a_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, {{Draw.Name}}Pipeline);
  {% endif %}

  const VkViewport viewport = { state.viewport.x, state.viewport.y, state.viewport.width, state.viewport.height, state.viewport.minDepth, state.viewport.maxDepth };
  const int32_t    x0       = std::max(int32_t(std::floor(state.viewport.x)), 0);
  const int32_t    y0       = std::max(int32_t(std::floor(state.viewport.y)), 0);
  const int32_t    x1       = std::min(int32_t(std::ceil(state.viewport.x + state.viewport.width)),  int32_t(m_gp_{{Draw.Pipeline}}.width));
  const int32_t    y1       = std::min(int32_t(std::ceil(state.viewport.y + state.viewport.height)), int32_t(m_gp_{{Draw.Pipeline}}.height));
  const VkRect2D   scissor  = { {x0, y0}, {uint32_t(std::max(x1 - x0, 0)), uint32_t(std::max(y1 - y0, 0))} };
  vkCmdSetViewport(a_commandBuffer, 0, 1, &viewport);
  vkCmdSetScissor (a_commandBuffer, 0, 1, &scissor);

  VkDescriptorSet descriptorSets[2] = {a_dsVS, a_dsPS};
  vkCmdBindDescriptorSets(a_commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, {{Draw.Name}}Layout, 0, {% if Draw.HasPS %}2{% else %}1{% endif %}, descriptorSets, 0, nullptr);

  {{Draw.Name}}_PC pcData;
  {% for Arg in Draw.UserArgs %}
  pcData.{{Arg.Name}} = {{Arg.Name}};
  {% endfor %}
  pcData.gpDummy = 0;
  vkCmdPushConstants(a_commandBuffer, {{Draw.Name}}Layout, VK_SHADER_STAGE_VERTEX_BIT, 0, uint32_t(sizeof(pcData)), &pcData);

  {% if Draw.Indexed %}
  vkCmdBindIndexBuffer(a_commandBuffer, a_indexBuffer, 0, VK_INDEX_TYPE_UINT32);
  {% if HasGPIndirect %}
  if(a_indirectId != GP_DIRECT_DRAW) // counts were written to m_gpIndirectBuffer by GPUpdateIndirectCmd
    vkCmdDrawIndexedIndirect(a_commandBuffer, m_gpIndirectBuffer, VkDeviceSize(a_indirectId)*8*sizeof(uint32_t), 1, 0);
  else
  {% endif %}
  vkCmdDrawIndexed(a_commandBuffer, a_count, {% if Draw.Instanced %}a_instNum{% else %}1{% endif %}, 0, 0, 0);
  {% else %}
  {% if HasGPIndirect %}
  if(a_indirectId != GP_DIRECT_DRAW) // counts were written to m_gpIndirectBuffer by GPUpdateIndirectCmd
    vkCmdDrawIndirect(a_commandBuffer, m_gpIndirectBuffer, VkDeviceSize(a_indirectId)*8*sizeof(uint32_t), 1, 0);
  else
  {% endif %}
  vkCmdDraw(a_commandBuffer, a_count, {% if Draw.Instanced %}a_instNum{% else %}1{% endif %}, 0, 0);
  {% endif %}
  {% if GPDynamicRendering %}
  vkCmdEndRendering(a_commandBuffer);
  {% else %}
  vkCmdEndRenderPass(a_commandBuffer);
  {% endif %}
}
{% endfor %}
