#include "points_render.h"
//#include "../utils/input_definitions.h"

#include <geom/vk_mesh.h>
#include <vk_pipeline.h>
#include <vk_buffers.h>
#include <vk_copy.h>

PointsRender::PointsRender(uint32_t a_width, uint32_t a_height) : m_width(a_width), m_height(a_height)
{
#ifdef NDEBUG
  m_enableValidation = false;
#else
  m_enableValidation = true;
#endif
}

void PointsRender::SetupDeviceFeatures()
{
   m_enabledDeviceFeatures.fillModeNonSolid = VK_TRUE;

   if(DISPLAY_MODE == RENDER_MODE::SPRITES)
     m_enabledDeviceFeatures.geometryShader = VK_TRUE;
}

void PointsRender::SetupDeviceExtensions()
{
  m_deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
  m_deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  m_deviceExtensions.push_back("VK_KHR_shader_float16_int8");
}

void PointsRender::SetupValidationLayers()
{
  m_validationLayers.push_back("VK_LAYER_KHRONOS_validation");
  m_validationLayers.push_back("VK_LAYER_LUNARG_monitor");
}

void
PointsRender::InitVulkan(const char **a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId)
{
  for (size_t i = 0; i < a_instanceExtensionsCount; ++i)
  {
    m_instanceExtensions.push_back(a_instanceExtensions[i]);
  }

  SetupValidationLayers();
  VK_CHECK_RESULT(volkInitialize());
  CreateInstance();
  volkLoadInstance(m_instance);

  CreateDevice(a_deviceId);
  volkLoadDevice(m_device);

  m_pCopy = std::make_shared<vk_utils::SimpleCopyHelper>(m_physicalDevice, m_device, m_transferQueue, m_queueFamilyIDXs.transfer,
                                                         8*1024*1024);

  m_commandPoolGraphics = vk_utils::createCommandPool(m_device, m_queueFamilyIDXs.graphics,
                                                      VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  m_commandPoolCompute = vk_utils::createCommandPool(m_device, m_queueFamilyIDXs.compute,
                                                     VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  m_pNBodySimGenerated = std::make_shared<nBody_GeneratedFix>();

  VkSemaphoreCreateInfo semaphoreInfo = {VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
  VK_CHECK_RESULT(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_simFinishedSem));

  m_cmdBuffersDrawMain.reserve(m_framesInFlight);
  m_cmdBuffersDrawMain = vk_utils::createCommandBuffers(m_device, m_commandPoolGraphics, m_framesInFlight);

  m_frameFences.resize(m_framesInFlight);
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < m_framesInFlight; i++)
  {
    VK_CHECK_RESULT(vkCreateFence(m_device, &fenceInfo, nullptr, &m_frameFences[i]));
  }
}

void PointsRender::InitPresentation(VkSurfaceKHR &a_surface)
{
  m_surface = a_surface;

  m_presentationResources.queue = m_swapchain.CreateSwapChain(m_physicalDevice, m_device, m_surface,
                                                              m_width, m_height, m_vsync);
  m_presentationResources.currentFrame = 0;

  VkSemaphoreCreateInfo semaphoreInfo = {};
  semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  VK_CHECK_RESULT(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_presentationResources.imageAvailable));
  VK_CHECK_RESULT(vkCreateSemaphore(m_device, &semaphoreInfo, nullptr, &m_presentationResources.renderingFinished));
  m_screenRenderPass = vk_utils::createDefaultRenderPass(m_device, m_swapchain.GetFormat());

  std::vector<VkFormat> depthFormats = {
      VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT,
      VK_FORMAT_D16_UNORM_S8_UINT,
      VK_FORMAT_D16_UNORM
  };
  vk_utils::getSupportedDepthFormat(m_physicalDevice, depthFormats, &m_depthBuffer.format);
  m_depthBuffer = vk_utils::createDepthTexture(m_device, m_physicalDevice, m_width, m_height, m_depthBuffer.format);
  m_frameBuffers = vk_utils::createFrameBuffers(m_device, m_swapchain, m_screenRenderPass, m_depthBuffer.view);

}

void PointsRender::CreateInstance()
{
  VkApplicationInfo appInfo = {};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pNext = nullptr;
  appInfo.pApplicationName = "VkRender";
  appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.pEngineName = "PointsRender";
  appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
  appInfo.apiVersion = VK_MAKE_VERSION(1, 1, 0);

  m_instance = vk_utils::createInstance(m_enableValidation, m_validationLayers, m_instanceExtensions, &appInfo);

  if (m_enableValidation)
    vk_utils::initDebugReportCallback(m_instance, &debugReportCallbackFn, &m_debugReportCallback);
}

void PointsRender::CreateDevice(uint32_t a_deviceId)
{
  SetupDeviceExtensions();
  m_physicalDevice = vk_utils::findPhysicalDevice(m_instance, true, a_deviceId, m_deviceExtensions);

  SetupDeviceFeatures();

  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;

  m_device = vk_utils::createLogicalDevice(m_physicalDevice, m_validationLayers, m_deviceExtensions,
                                           m_enabledDeviceFeatures, m_queueFamilyIDXs,
                                           VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT,
                                           &features);

  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.graphics, 0, &m_graphicsQueue);
  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.transfer, 0, &m_transferQueue);
  vkGetDeviceQueue(m_device, m_queueFamilyIDXs.compute, 0, &m_computeQueue);
}

void PointsRender::SetupPointsVertexBindings()
{
  m_pointsData.inputBinding.binding = 0;
  m_pointsData.inputBinding.stride = sizeof(float) * 8;
  m_pointsData.inputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

  m_pointsData.inputAttributes[0].binding = 0;
  m_pointsData.inputAttributes[0].location = 0;
  m_pointsData.inputAttributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_pointsData.inputAttributes[0].offset = 0;

  m_pointsData.inputAttributes[1].binding = 0;
  m_pointsData.inputAttributes[1].location = 1;
  m_pointsData.inputAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
  m_pointsData.inputAttributes[1].offset = sizeof(float) * 4;

  m_pointsData.inputStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  m_pointsData.inputStateInfo.vertexBindingDescriptionCount = 1;
  m_pointsData.inputStateInfo.vertexAttributeDescriptionCount =
      sizeof(m_pointsData.inputAttributes) / sizeof(m_pointsData.inputAttributes[0]);
  m_pointsData.inputStateInfo.pVertexBindingDescriptions = &m_pointsData.inputBinding;
  m_pointsData.inputStateInfo.pVertexAttributeDescriptions = m_pointsData.inputAttributes;
}

void PointsRender::SetupPointsPipeline()
{
  SetupPointsVertexBindings();

  std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1}
  };
  m_pBindings = std::make_shared<vk_utils::DescriptorMaker>(m_device, dtypes, 1);

  m_pBindings->BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pBindings->BindImage(0, m_colormap.view, m_colormapSampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  m_pBindings->BindEnd(&m_dSet, &m_dSetLayout);

  vk_utils::GraphicsPipelineMaker maker;

  std::unordered_map<VkShaderStageFlagBits, std::string> shader_paths;
  shader_paths[VK_SHADER_STAGE_VERTEX_BIT] = "shaders/points.vert.spv";
  shader_paths[VK_SHADER_STAGE_FRAGMENT_BIT] = "shaders/points.frag.spv";


  maker.LoadShaders(m_device, shader_paths);

  m_pointsPipeline.layout = maker.MakeLayout(m_device, {m_dSetLayout}, sizeof(m_pushConsts));
  maker.SetDefaultState(m_width, m_height);
  maker.rasterizer.polygonMode = VK_POLYGON_MODE_POINT;
  m_pointsPipeline.pipeline = maker.MakePipeline(m_device, m_pointsData.inputStateInfo,
                                                 m_screenRenderPass,
                                                 {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR},
                                                 vk_utils::IA_PList());
}

void PointsRender::SetupSpritesPipeline()
{
  SetupPointsVertexBindings();

  std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2}
  };
  m_pBindings = std::make_shared<vk_utils::DescriptorMaker>(m_device, dtypes, 1);

  m_pBindings->BindBegin(VK_SHADER_STAGE_FRAGMENT_BIT);
  m_pBindings->BindImage(0, m_sprite.view, m_spriteSampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  m_pBindings->BindImage(1, m_colormap.view, m_colormapSampler, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
  m_pBindings->BindEnd(&m_dSet, &m_dSetLayout);

  vk_utils::GraphicsPipelineMaker maker;

  std::unordered_map<VkShaderStageFlagBits, std::string> shader_paths;
  shader_paths[VK_SHADER_STAGE_VERTEX_BIT] = "shaders/points.vert.spv";
  shader_paths[VK_SHADER_STAGE_GEOMETRY_BIT] = "shaders/points.geom.spv";
  shader_paths[VK_SHADER_STAGE_FRAGMENT_BIT] = "shaders/sprites.frag.spv";

  maker.LoadShaders(m_device, shader_paths);

  m_pointsPipeline.layout = maker.MakeLayout(m_device, {m_dSetLayout}, sizeof(m_pushConsts));
  maker.SetDefaultState(m_width, m_height);

  maker.colorBlendAttachments[0].blendEnable = VK_TRUE;
  maker.colorBlendAttachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  maker.colorBlendAttachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  maker.colorBlendAttachments[0].colorBlendOp = VK_BLEND_OP_ADD;
  maker.colorBlendAttachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  maker.colorBlendAttachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  maker.colorBlendAttachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
  maker.colorBlendAttachments[0].colorWriteMask = vk_utils::ALL_COLOR_COMPONENTS;
  maker.depthStencilTest.depthWriteEnable = VK_FALSE;
  m_pointsPipeline.pipeline = maker.MakePipeline(m_device, m_pointsData.inputStateInfo,
                                                 m_screenRenderPass,
                                                 {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR},
                                                 vk_utils::IA_PList());
}

void PointsRender::CreateUniformBuffer()
{
//  VkMemoryRequirements memReq;
//  m_ubo = vk_utils::createBuffer(m_device, sizeof(UniformParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, &memReq);
//
//  VkMemoryAllocateInfo allocateInfo = {};
//  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
//  allocateInfo.pNext = nullptr;
//  allocateInfo.allocationSize = memReq.size;
//  allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memReq.memoryTypeBits,
//                                                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
//                                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
//                                                          m_physicalDevice);
//  VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, nullptr, &m_uboAlloc));
//
//  VK_CHECK_RESULT(vkBindBufferMemory(m_device, m_ubo, m_uboAlloc, 0));
//
//  vkMapMemory(m_device, m_uboAlloc, 0, sizeof(m_uniforms), 0, &m_uboMappedMem);
//
//  m_uniforms.lightPos = LiteMath::float3(0.0f, 1.0f, 1.0f);
//  m_uniforms.baseColor = LiteMath::float3(0.9f, 0.92f, 1.0f);
//  m_uniforms.animateLightColor = true;
//
//  UpdateUniformBuffer(0.0f);
}

void PointsRender::UpdateUniformBuffer(float a_time)
{
  // most uniforms are updated in GUI -> SetupGUIElements()
//  m_uniforms.time = a_time;
//  memcpy(m_uboMappedMem, &m_uniforms, sizeof(m_uniforms));
}

void PointsRender::BuildDrawCommandBuffer(VkCommandBuffer a_cmdBuff, VkFramebuffer a_frameBuff, VkPipeline a_pipeline)
{
  vkResetCommandBuffer(a_cmdBuff, 0);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

  VK_CHECK_RESULT(vkBeginCommandBuffer(a_cmdBuff, &beginInfo));

  vk_utils::setDefaultViewport(a_cmdBuff, static_cast<float>(m_width), static_cast<float>(m_height));
  vk_utils::setDefaultScissor(a_cmdBuff, m_width, m_height);

  {
    VkRenderPassBeginInfo renderPassInfo = {};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = m_screenRenderPass;
    renderPassInfo.framebuffer = a_frameBuff;
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = m_swapchain.GetExtent();

    VkClearValue clearValues[2] = {};
    clearValues[0].color = {0.0f, 0.0f, 0.0f, 1.0f};
    clearValues[1].depthStencil = {1.0f, 0};
    renderPassInfo.clearValueCount = 2;
    renderPassInfo.pClearValues = &clearValues[0];

    vkCmdBeginRenderPass(a_cmdBuff, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, a_pipeline);

    VkShaderStageFlags stageFlags = (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    if(DISPLAY_MODE == RENDER_MODE::SPRITES)
    {
      stageFlags |= VK_SHADER_STAGE_GEOMETRY_BIT;
    }
    vkCmdBindDescriptorSets(a_cmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pointsPipeline.layout, 0, 1,
                            &m_dSet, 0, VK_NULL_HANDLE);

    VkDeviceSize zero_offset = 0u;
    VkBuffer vertexBuf = m_pointsData.pointsBuf;

    vkCmdBindVertexBuffers(a_cmdBuff, 0, 1, &vertexBuf, &zero_offset);
    vkCmdPushConstants(a_cmdBuff, m_pointsPipeline.layout, stageFlags, 0,
                       sizeof(m_pushConsts), &m_pushConsts);

    vkCmdDraw(a_cmdBuff, m_pointsData.pointsCount, 1, 0, 0);

    vkCmdEndRenderPass(a_cmdBuff);
  }

  VK_CHECK_RESULT(vkEndCommandBuffer(a_cmdBuff));
}


void PointsRender::CleanupPipelineAndSwapchain()
{
  if (!m_cmdBuffersDrawMain.empty())
  {
    vkFreeCommandBuffers(m_device, m_commandPoolGraphics, static_cast<uint32_t>(m_cmdBuffersDrawMain.size()),
                         m_cmdBuffersDrawMain.data());
    m_cmdBuffersDrawMain.clear();
  }

  for (size_t i = 0; i < m_frameFences.size(); i++)
  {
    vkDestroyFence(m_device, m_frameFences[i], nullptr);
  }

  vkDestroyImageView(m_device, m_depthBuffer.view, nullptr);
  vkDestroyImage(m_device, m_depthBuffer.image, nullptr);

  for (size_t i = 0; i < m_frameBuffers.size(); i++)
  {
    vkDestroyFramebuffer(m_device, m_frameBuffers[i], nullptr);
  }

  vkDestroyRenderPass(m_device, m_screenRenderPass, nullptr);

  //  m_swapchain.Cleanup();
}

void PointsRender::RecreateSwapChain()
{
  vkDeviceWaitIdle(m_device);

  CleanupPipelineAndSwapchain();
  m_presentationResources.queue = m_swapchain.CreateSwapChain(m_physicalDevice, m_device, m_surface, m_width, m_height,
                                                              m_vsync);
  std::vector<VkFormat> depthFormats = {
      VK_FORMAT_D32_SFLOAT,
      VK_FORMAT_D32_SFLOAT_S8_UINT,
      VK_FORMAT_D24_UNORM_S8_UINT,
      VK_FORMAT_D16_UNORM_S8_UINT,
      VK_FORMAT_D16_UNORM
  };
  vk_utils::getSupportedDepthFormat(m_physicalDevice, depthFormats, &m_depthBuffer.format);

  m_screenRenderPass = vk_utils::createDefaultRenderPass(m_device, m_swapchain.GetFormat());
  m_depthBuffer = vk_utils::createDepthTexture(m_device, m_physicalDevice, m_width, m_height, m_depthBuffer.format);
  m_frameBuffers = vk_utils::createFrameBuffers(m_device, m_swapchain, m_screenRenderPass, m_depthBuffer.view);

  m_frameFences.resize(m_framesInFlight);
  VkFenceCreateInfo fenceInfo = {};
  fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (size_t i = 0; i < m_framesInFlight; i++)
  {
    VK_CHECK_RESULT(vkCreateFence(m_device, &fenceInfo, nullptr, &m_frameFences[i]));
  }

  m_cmdBuffersDrawMain = vk_utils::createCommandBuffers(m_device, m_commandPoolGraphics, m_framesInFlight);
  for (size_t i = 0; i < m_swapchain.GetImageCount(); ++i)
  {
    BuildDrawCommandBuffer(m_cmdBuffersDrawMain[i], m_frameBuffers[i], m_pointsPipeline.pipeline);
  }
}

void PointsRender::Cleanup()
{
  CleanupPipelineAndSwapchain();

  if (m_pointsPipeline.pipeline != VK_NULL_HANDLE)
  {
    vkDestroyPipeline(m_device, m_pointsPipeline.pipeline, nullptr);
  }
  if (m_pointsPipeline.layout != VK_NULL_HANDLE)
  {
    vkDestroyPipelineLayout(m_device, m_pointsPipeline.layout, nullptr);
  }

  if (m_presentationResources.imageAvailable != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(m_device, m_presentationResources.imageAvailable, nullptr);
  }
  if (m_presentationResources.renderingFinished != VK_NULL_HANDLE)
  {
    vkDestroySemaphore(m_device, m_presentationResources.renderingFinished, nullptr);
  }

  if (m_commandPoolGraphics != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(m_device, m_commandPoolGraphics, nullptr);
  }
}

void PointsRender::ProcessInput(const AppInput &input)
{
  // add keyboard controls here
  // camera movement is processed separately
  //  if(input.keyPressed[GLFW_KEY_SPACE])
  //
}

void PointsRender::UpdateCamera(const Camera *cams, uint32_t a_camsCount)
{
  assert(a_camsCount > 0);
  m_cam = cams[0];
  UpdateView();
}

void PointsRender::UpdateView()
{
  const float aspect = float(m_width) / float(m_height);
  auto mProjFix = OpenglToVulkanProjectionMatrixFix();
  auto mProj = projectionMatrix(m_cam.fov, aspect, 0.1f, 1000.0f);
  auto mLookAt = LiteMath::lookAt(m_cam.pos, m_cam.lookAt, m_cam.up);
  auto mWorldViewProj = mProjFix * mProj * mLookAt;
  m_pushConsts.projView = mWorldViewProj;
  m_pushConsts.cameraPos = LiteMath::float4(m_cam.pos.x, m_cam.pos.y, m_cam.pos.z, 1.0f);
}

void PointsRender::DrawFrameSimple()
{
  vkWaitForFences(m_device, 1, &m_frameFences[m_presentationResources.currentFrame], VK_TRUE, UINT64_MAX);
  vkResetFences(m_device, 1, &m_frameFences[m_presentationResources.currentFrame]);

  uint32_t imageIdx;
  m_swapchain.AcquireNextImage(m_presentationResources.imageAvailable, &imageIdx);

  auto currentCmdBuf = m_cmdBuffersDrawMain[imageIdx];

  VkSemaphore waitSemaphores[] = {m_presentationResources.imageAvailable};
  VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};

  BuildDrawCommandBuffer(currentCmdBuf, m_frameBuffers[imageIdx], m_pointsPipeline.pipeline);

  auto simCmdBuf = BuildCommandBufferSimulation();

  VkSubmitInfo simSubmit = {};
  simSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  simSubmit.waitSemaphoreCount = 1;
  simSubmit.pWaitSemaphores = waitSemaphores;
  simSubmit.pWaitDstStageMask = waitStages;
  simSubmit.commandBufferCount = 1;
  simSubmit.pCommandBuffers = &simCmdBuf;
  simSubmit.signalSemaphoreCount = 1;
  simSubmit.pSignalSemaphores = &m_simFinishedSem;

  VK_CHECK_RESULT(vkQueueSubmit(m_computeQueue, 1, &simSubmit, VK_NULL_HANDLE));

  VkPipelineStageFlags drawWaitStages[] = {VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT};

  VkSubmitInfo submitInfo = {};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &m_simFinishedSem;
  submitInfo.pWaitDstStageMask = drawWaitStages;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &currentCmdBuf;

  VkSemaphore signalSemaphores[] = {m_presentationResources.renderingFinished};
  submitInfo.signalSemaphoreCount = 1;
  submitInfo.pSignalSemaphores = signalSemaphores;

  VK_CHECK_RESULT(vkQueueSubmit(m_graphicsQueue, 1, &submitInfo, m_frameFences[m_presentationResources.currentFrame]));

  VkResult presentRes = m_swapchain.QueuePresent(m_presentationResources.queue, imageIdx,
                                                 m_presentationResources.renderingFinished);

  if (presentRes == VK_ERROR_OUT_OF_DATE_KHR || presentRes == VK_SUBOPTIMAL_KHR)
  {
    RecreateSwapChain();
  }
  else
    if (presentRes != VK_SUCCESS)
    {
      RUN_TIME_ERROR("Failed to present swapchain image");
    }

  m_presentationResources.currentFrame = (m_presentationResources.currentFrame + 1) % m_framesInFlight;

  vkQueueWaitIdle(m_presentationResources.queue);
}

void PointsRender::DrawFrame(float a_time)
{
  UpdateUniformBuffer(a_time);
  DrawFrameSimple();
}
