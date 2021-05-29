#ifdef WIN32
#pragma comment(lib,"glfw3.lib")
#endif

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <thread>

#include "hello_tri.h"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

const int WIDTH  = 800;
const int HEIGHT = 600;

const int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<const char*> deviceExtensions = {
  VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class HelloTriangleApplication : public IDrawFrameApp
{
public:

  HelloTriangleApplication() { }
  ~HelloTriangleApplication(){ Cleanup(); }
  
  void DoFrame() override
  {
    glfwPollEvents();
    DrawFrame();
  }

  void Init(VkInstance instance, VkPhysicalDevice a_physicalDevice, VkDevice a_device) override 
  {
    if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
      throw std::runtime_error("glfwCreateWindowSurface: failed to create window surface!");

    InitVulkanObjects(a_physicalDevice, a_device);
    CreateResources();
  }
  
  void InitWindow() override
  {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
  }
  
  //void run(VkPhysicalDevice a_physicalDevice, VkDevice a_device) 
  //{
  //  InitWindow();
  //  InitVulkanObjects(a_physicalDevice, a_device);
  //  CreateResources();
  //  MainLoop();
  //  Cleanup();
  //}

private:

  GLFWwindow * window;

  VkInstance instance;
  std::vector<const char*> enabledLayers;

  VkDebugUtilsMessengerEXT debugMessenger;
  VkSurfaceKHR surface;

  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice device;

  VkQueue graphicsQueue;
  VkQueue presentQueue;

  vk_utils::ScreenBufferResources screen;

  VkRenderPass     renderPass;
  VkPipelineLayout pipelineLayout;
  VkPipeline       graphicsPipeline;

  VkCommandPool                commandPool;
  std::vector<VkCommandBuffer> commandBuffers;

  VkBuffer       m_vbo;     //  
  VkDeviceMemory m_vboMem;  // we will store our vertices data here

  struct SyncObj
  {
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence>     inFlightFences;
  } m_sync;

  size_t currentFrame = 0;

  static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
    VkDebugReportFlagsEXT                       flags,
    VkDebugReportObjectTypeEXT                  objectType,
    uint64_t                                    object,
    size_t                                      location,
    int32_t                                     messageCode,
    const char*                                 pLayerPrefix,
    const char*                                 pMessage,
    void*                                       pUserData)
  {
    printf("[Debug Report]: %s: %s\n", pLayerPrefix, pMessage);
    return VK_FALSE;
  }
  VkDebugReportCallbackEXT debugReportCallback;
  

  void InitVulkanObjects(VkPhysicalDevice a_physicalDevice, VkDevice a_device) 
  {
    physicalDevice = a_physicalDevice;
    device         = a_device;

    auto queueFID  = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT);

    VkBool32 presentSupport = false;
    vkGetPhysicalDeviceSurfaceSupportKHR(physicalDevice, queueFID, surface, &presentSupport);
    if (!presentSupport)
      throw std::runtime_error("vkGetPhysicalDeviceSurfaceSupportKHR: no present support for the target device and graphics queue");

    vkGetDeviceQueue(device, queueFID, 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFID, 0, &presentQueue);
    
    // ==> commadnPool
    {
      VkCommandPoolCreateInfo poolInfo = {};
      poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      poolInfo.queueFamilyIndex = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT);

      if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS)
        throw std::runtime_error("[CreateCommandPoolAndBuffers]: failed to create command pool!");
    }
    
    VkSurfaceTransformFlagsKHR preTransf = {};
    uint32_t w = WIDTH, h = HEIGHT;
    screen.swapchain.CreateSwapChain(physicalDevice, device, surface, w, h, false, preTransf);
    //vk_utils::CreateScreenImageViews(device, &screen);
  }

  void CreateResources()
  {
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    CreateRenderPass(device, screen.swapchain.GetFormat(), 
                     &renderPass);

    CreateGraphicsPipeline(device, screen.swapchain.GetExtent(), renderPass, 
                           &pipelineLayout, &graphicsPipeline);
  
    //CreateScreenFrameBuffers(device, renderPass, &screen);
    vk_utils::CreateScreenFrameBuffers(device, renderPass, screen.depthImageView, 
                                       &screen);

    CreateVertexBuffer(device, physicalDevice, 6*2*sizeof(float),
                       &m_vbo, &m_vboMem);

    CreateAndWriteCommandBuffers(device, commandPool, screen.framebuffers, screen.swapchain.GetExtent(), renderPass, graphicsPipeline, m_vbo,
                                 &commandBuffers);

    CreateSyncObjects(device, &m_sync);

   
    // put our vertices to GPU
    //
    float trianglePos[] =
    {
      -0.5f, -0.5f,
      0.5f, -0.5f,
      0.0f, +0.5f,
    };

    PutTriangleVerticesToVBO_Now(device, commandPool, graphicsQueue, trianglePos, 6*2,
                                 m_vbo);
  }

  void MainLoop()
  {
    while (!glfwWindowShouldClose(window)) 
    {
      glfwPollEvents();
      DrawFrame();
    }

    vkDeviceWaitIdle(device);
  }

  void Cleanup() 
  { 
    // free our vbo
    vkFreeMemory(device, m_vboMem, NULL);
    vkDestroyBuffer(device, m_vbo, NULL);

    if (enableValidationLayers)
    {
      //// destroy callback.
      //auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
      //if (func == nullptr)
      //  throw std::runtime_error("Could not load vkDestroyDebugReportCallbackEXT");
      //func(instance, debugReportCallback, NULL);
    }

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
      vkDestroySemaphore(device, m_sync.renderFinishedSemaphores[i], nullptr);
      vkDestroySemaphore(device, m_sync.imageAvailableSemaphores[i], nullptr);
      vkDestroyFence    (device, m_sync.inFlightFences[i], nullptr);
    }

    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyPipeline      (device, graphicsPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass    (device, renderPass, nullptr);

    screen.swapchain.Cleanup();
    vkDestroyDevice(device, nullptr);

    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);

    glfwDestroyWindow(window);

    glfwTerminate();
  }

  static void CreateRenderPass(VkDevice a_device, VkFormat a_swapChainImageFormat,
                               VkRenderPass* a_pRenderPass)
  {
    VkAttachmentDescription colorAttachment = {};
    colorAttachment.format         = a_swapChainImageFormat;
    colorAttachment.samples        = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference colorAttachmentRef = {};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass  = {};
    subpass.pipelineBindPoint     = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount  = 1;
    subpass.pColorAttachments     = &colorAttachmentRef;

    VkSubpassDependency dependency = {};
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0;
    dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo = {};
    renderPassInfo.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments    = &colorAttachment;
    renderPassInfo.subpassCount    = 1;
    renderPassInfo.pSubpasses      = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies   = &dependency;

    if (vkCreateRenderPass(a_device, &renderPassInfo, nullptr, a_pRenderPass) != VK_SUCCESS)
      throw std::runtime_error("[CreateRenderPass]: failed to create render pass!");
  }

  static void CreateGraphicsPipeline(VkDevice a_device, VkExtent2D a_screenExtent, VkRenderPass a_renderPass,
                                     VkPipelineLayout* a_pLayout, VkPipeline* a_pPipiline)
  {
    auto vertShaderCode = vk_utils::ReadFile("shaders/vert.spv");
    auto fragShaderCode = vk_utils::ReadFile("shaders/frag.spv");

    VkShaderModule vertShaderModule = vk_utils::CreateShaderModule(a_device, vertShaderCode);
    VkShaderModule fragShaderModule = vk_utils::CreateShaderModule(a_device, fragShaderCode);

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

    VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };


    VkVertexInputBindingDescription vInputBinding = { };
    vInputBinding.binding   = 0;
    vInputBinding.stride    = sizeof(float) * 2;
    vInputBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription vAttribute = {};
    vAttribute.binding  = 0;
    vAttribute.location = 0;
    vAttribute.format   = VK_FORMAT_R32G32_SFLOAT;
    vAttribute.offset   = 0;

    VkPipelineVertexInputStateCreateInfo vertexInputInfo = {};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInputInfo.vertexBindingDescriptionCount   = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions      = &vInputBinding;
    vertexInputInfo.pVertexAttributeDescriptions    = &vAttribute;
    
    VkPipelineInputAssemblyStateCreateInfo inputAssembly = {};
    inputAssembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport = {};
    viewport.x        = 0.0f;
    viewport.y        = 0.0f;
    viewport.width    = (float)a_screenExtent.width;
    viewport.height   = (float)a_screenExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor = {};
    scissor.offset = { 0, 0 };
    scissor.extent = a_screenExtent;

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

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount         = 0;
    pipelineLayoutInfo.pushConstantRangeCount = 0;

    if (vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, a_pLayout) != VK_SUCCESS)
      throw std::runtime_error("[CreateGraphicsPipeline]: failed to create pipeline layout!");

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
    pipelineInfo.layout              = (*a_pLayout);
    pipelineInfo.renderPass          = a_renderPass;
    pipelineInfo.subpass             = 0;
    pipelineInfo.basePipelineHandle  = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(a_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, a_pPipiline) != VK_SUCCESS)
      throw std::runtime_error("[CreateGraphicsPipeline]: failed to create graphics pipeline!");

    vkDestroyShaderModule(a_device, fragShaderModule, nullptr);
    vkDestroyShaderModule(a_device, vertShaderModule, nullptr);
  }


  static void CreateAndWriteCommandBuffers(VkDevice a_device, VkCommandPool a_cmdPool, std::vector<VkFramebuffer> a_swapChainFramebuffers, VkExtent2D a_frameBufferExtent,
                                           VkRenderPass a_renderPass, VkPipeline a_graphicsPipeline, VkBuffer a_vPosBuffer,
                                           std::vector<VkCommandBuffer>* a_cmdBuffers) 
  {
    std::vector<VkCommandBuffer>& commandBuffers = (*a_cmdBuffers);

    commandBuffers.resize(a_swapChainFramebuffers.size());

    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = a_cmdPool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

    if (vkAllocateCommandBuffers(a_device, &allocInfo, commandBuffers.data()) != VK_SUCCESS)
      throw std::runtime_error("[CreateCommandPoolAndBuffers]: failed to allocate command buffers!");

    for (size_t i = 0; i < commandBuffers.size(); i++) 
    {
      VkCommandBufferBeginInfo beginInfo = {};
      beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

      if (vkBeginCommandBuffer(commandBuffers[i], &beginInfo) != VK_SUCCESS) 
        throw std::runtime_error("[CreateCommandPoolAndBuffers]: failed to begin recording command buffer!");

      VkRenderPassBeginInfo renderPassInfo = {};
      renderPassInfo.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
      renderPassInfo.renderPass        = a_renderPass;
      renderPassInfo.framebuffer       = a_swapChainFramebuffers[i];
      renderPassInfo.renderArea.offset = { 0, 0 };
      renderPassInfo.renderArea.extent = a_frameBufferExtent;

      VkClearValue clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
      renderPassInfo.clearValueCount = 1;
      renderPassInfo.pClearValues = &clearColor;

      vkCmdBeginRenderPass(commandBuffers[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

      vkCmdBindPipeline(commandBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, a_graphicsPipeline);

      // say we want to take vertices pos from a_vPosBuffer
      {
        VkBuffer vertexBuffers[] = { a_vPosBuffer };
        VkDeviceSize offsets[]   = { 0 };
        vkCmdBindVertexBuffers(commandBuffers[i], 0, 1, vertexBuffers, offsets);
      }

      vkCmdDraw(commandBuffers[i], 3, 1, 0, 0);

      vkCmdEndRenderPass(commandBuffers[i]);

      if (vkEndCommandBuffer(commandBuffers[i]) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
      }
    }
  }

  static void CreateSyncObjects(VkDevice a_device, SyncObj* a_pSyncObjs)
  {
    a_pSyncObjs->imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    a_pSyncObjs->renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    a_pSyncObjs->inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

    VkSemaphoreCreateInfo semaphoreInfo = {};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo = {};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
    {
      if (vkCreateSemaphore(a_device, &semaphoreInfo, nullptr, &a_pSyncObjs->imageAvailableSemaphores[i]) != VK_SUCCESS ||
          vkCreateSemaphore(a_device, &semaphoreInfo, nullptr, &a_pSyncObjs->renderFinishedSemaphores[i]) != VK_SUCCESS ||
          vkCreateFence    (a_device, &fenceInfo,     nullptr, &a_pSyncObjs->inFlightFences[i]) != VK_SUCCESS) {
        throw std::runtime_error("[CreateSyncObjects]: failed to create synchronization objects for a frame!");
      }
    }
  }

  static void CreateVertexBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                                 VkBuffer *a_pBuffer, VkDeviceMemory *a_pBufferMemory)
  {
   
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.pNext       = nullptr;
    bufferCreateInfo.size        = a_bufferSize;                         
    bufferCreateInfo.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;            

    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, a_pBuffer)); // create bufferStaging.

                
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(a_device, (*a_pBuffer), &memoryRequirements);


    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = memoryRequirements.size; // specify required memory.
    allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memoryRequirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice); // #NOTE VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT

    VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, a_pBufferMemory));   // allocate memory on device.

    VK_CHECK_RESULT(vkBindBufferMemory(a_device, (*a_pBuffer), (*a_pBufferMemory), 0));  // Now associate that allocated memory with the bufferStaging. With that, the bufferStaging is backed by actual memory.
  }

  static void RunCommandBuffer(VkCommandBuffer a_cmdBuff, VkQueue a_queue, VkDevice a_device)
  {
    // Now we shall finally submit the recorded command bufferStaging to a queue.
    //
    VkSubmitInfo submitInfo = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1; // submit a single command bufferStaging
    submitInfo.pCommandBuffers    = &a_cmdBuff; // the command bufferStaging to submit.
                                         
    VkFence fence;
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));

    // We submit the command bufferStaging on the queue, at the same time giving a fence.
    //
    VK_CHECK_RESULT(vkQueueSubmit(a_queue, 1, &submitInfo, fence));

    // The command will not have finished executing until the fence is signalled.
    // So we wait here. We will directly after this read our bufferStaging from the GPU,
    // and we will not be sure that the command has finished executing unless we wait for the fence.
    // Hence, we use a fence here.
    //
    VK_CHECK_RESULT(vkWaitForFences(a_device, 1, &fence, VK_TRUE, 100000000000));

    vkDestroyFence(a_device, fence, NULL);
  }

  // An example function that immediately copy vertex data to GPU
  //
  static void PutTriangleVerticesToVBO_Now(VkDevice a_device, VkCommandPool a_pool, VkQueue a_queue, float* a_triPos, int a_floatsNum,
                                           VkBuffer a_buffer)
  {
    VkCommandBufferAllocateInfo allocInfo = {};
    allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool        = a_pool;
    allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmdBuff;
    if (vkAllocateCommandBuffers(a_device, &allocInfo, &cmdBuff) != VK_SUCCESS)
      throw std::runtime_error("[PutTriangleVerticesToVBO_Now]: failed to allocate command buffer!");

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT; 
    
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
    vkCmdUpdateBuffer   (cmdBuff, a_buffer, 0, a_floatsNum * sizeof(float), a_triPos);
    vkEndCommandBuffer  (cmdBuff);

    RunCommandBuffer(cmdBuff, a_queue, a_device);

    vkFreeCommandBuffers(a_device, a_pool, 1, &cmdBuff);
  }


  void DrawFrame() 
  {
    vkWaitForFences(device, 1, &m_sync.inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences  (device, 1, &m_sync.inFlightFences[currentFrame]);

    uint32_t imageIndex;
    vkAcquireNextImageKHR(device, screen.swapchain.GetVulkanSwapChain(), UINT64_MAX, m_sync.imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

    VkSemaphore      waitSemaphores[] = { m_sync.imageAvailableSemaphores[currentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };

    VkSubmitInfo submitInfo = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores    = waitSemaphores;
    submitInfo.pWaitDstStageMask  = waitStages;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &commandBuffers[imageIndex];

    VkSemaphore signalSemaphores[]  = { m_sync.renderFinishedSemaphores[currentFrame] };
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores    = signalSemaphores;

    if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, m_sync.inFlightFences[currentFrame]) != VK_SUCCESS)
      throw std::runtime_error("[DrawFrame]: failed to submit draw command buffer!");

    VkPresentInfoKHR presentInfo = {};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores    = signalSemaphores;

    VkSwapchainKHR swapChains[] = { screen.swapchain.GetVulkanSwapChain() };
    presentInfo.swapchainCount  = 1;
    presentInfo.pSwapchains     = swapChains;
    presentInfo.pImageIndices   = &imageIndex;

    vkQueuePresentKHR(presentQueue, &presentInfo);
    currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
  }

};

std::shared_ptr<IDrawFrameApp> CreateHelloTriImpl()
{
  return std::make_shared<HelloTriangleApplication>();
}
