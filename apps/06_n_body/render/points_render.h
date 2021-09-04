#ifndef CHIMERA_SIMPLE_RENDER_H
#define CHIMERA_SIMPLE_RENDER_H

#define VK_NO_PROTOTYPES

#include "render_common.h"
#include <geom/vk_mesh.h>
#include <vk_descriptor_sets.h>
#include <vk_copy.h>
#include <vk_images.h>
#include <vk_swapchain.h>
#include <string>
#include <iostream>
#include <generated_userfix.h>

class PointsRender : public IRender
{
public:
  PointsRender(uint32_t a_width, uint32_t a_height);

  ~PointsRender()
  { Cleanup(); };

  inline uint32_t GetWidth() const override
  { return m_width; }

  inline uint32_t GetHeight() const override
  { return m_height; }

  inline VkInstance GetVkInstance() const override
  { return m_instance; }

  void InitVulkan(const char **a_instanceExtensions, uint32_t a_instanceExtensionsCount, uint32_t a_deviceId) override;

  void InitPresentation(VkSurfaceKHR &a_surface) override;

  void ProcessInput(const AppInput &input) override;

  void UpdateCamera(const Camera *cams, uint32_t a_camsCount) override;

  void UpdateView();

  void LoadScene(const char* path, bool transpose_inst_matrices) override;

  void DrawFrame(float a_time) override;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  // debugging utils
  //
  static VKAPI_ATTR VkBool32 VKAPI_CALL debugReportCallbackFn(
      VkDebugReportFlagsEXT flags,
      VkDebugReportObjectTypeEXT objectType,
      uint64_t object,
      size_t location,
      int32_t messageCode,
      const char *pLayerPrefix,
      const char *pMessage,
      void *pUserData)
  {
    std::cout << pLayerPrefix << ": " << pMessage << std::endl;
    return VK_FALSE;
  }

  VkDebugReportCallbackEXT m_debugReportCallback = nullptr;
private:

  VkInstance m_instance = VK_NULL_HANDLE;
  VkCommandPool m_commandPoolGraphics = VK_NULL_HANDLE;
  VkCommandPool m_commandPoolCompute = VK_NULL_HANDLE;
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  VkQueue m_graphicsQueue = VK_NULL_HANDLE;
  VkQueue m_transferQueue = VK_NULL_HANDLE;
  VkQueue m_computeQueue = VK_NULL_HANDLE;

  std::shared_ptr<vk_utils::ICopyEngine> m_pCopy;

  vk_utils::QueueFID_T m_queueFamilyIDXs{UINT32_MAX, UINT32_MAX, UINT32_MAX};

  struct
  {
    uint32_t currentFrame = 0u;
    VkQueue queue = VK_NULL_HANDLE;
    VkSemaphore imageAvailable = VK_NULL_HANDLE;
    VkSemaphore renderingFinished = VK_NULL_HANDLE;
  } m_presentationResources;

  std::vector<VkFence> m_frameFences;
  std::vector<VkCommandBuffer> m_cmdBuffersDrawMain;

  struct
  {
    LiteMath::float4x4 projView;
    LiteMath::float4x4 model;
  } pushConst2M;

//  UniformParams m_uniforms{};
//  VkBuffer m_ubo = VK_NULL_HANDLE;
//  VkDeviceMemory m_uboAlloc = VK_NULL_HANDLE;
//  void *m_uboMappedMem = nullptr;

  std::shared_ptr<nBody_GeneratedFix> m_pNBodySimGenerated;
  VkSemaphore m_simFinishedSem;
  struct
  {
    VkVertexInputBindingDescription   inputBinding {};
    VkVertexInputAttributeDescription inputAttributes[2] {};
    VkPipelineVertexInputStateCreateInfo inputStateInfo {};
    VkBuffer pointsBuf = VK_NULL_HANDLE;
    VkDeviceMemory pointsMem = VK_NULL_HANDLE;
    uint32_t pointsCount = nBody::BODIES_COUNT;
    std::vector<nBody::BodyState> outBodies;
  } m_pointsData;

  struct pipeline_data_t
  {
    VkPipelineLayout layout;
    VkPipeline pipeline;
  };
  pipeline_data_t m_pointsPipeline {};

  VkDescriptorSet m_dSet = VK_NULL_HANDLE;
  VkDescriptorSetLayout m_dSetLayout = VK_NULL_HANDLE;
  VkRenderPass m_screenRenderPass = VK_NULL_HANDLE; // main renderpass

  std::shared_ptr<vk_utils::DescriptorMaker> m_pBindings = nullptr;

  // *** presentation
  VkSurfaceKHR m_surface = VK_NULL_HANDLE;
  VulkanSwapChain m_swapchain;
  std::vector<VkFramebuffer> m_frameBuffers;
  vk_utils::VulkanImageMem m_depthBuffer{};
  // ***

  Camera m_cam;
  uint32_t m_width = 1024u;
  uint32_t m_height = 1024u;
  uint32_t m_framesInFlight = 2u;
  bool m_vsync = false;

  VkPhysicalDeviceFeatures m_enabledDeviceFeatures = {};
  std::vector<const char *> m_deviceExtensions = {};
  std::vector<const char *> m_instanceExtensions = {};

  bool m_enableValidation;
  std::vector<const char *> m_validationLayers;

  void DrawFrameSimple();
  void CreateInstance();
  void CreateDevice(uint32_t a_deviceId);
  void BuildCommandBufferPoints(VkCommandBuffer cmdBuff, VkFramebuffer frameBuff, VkPipeline a_pipeline);
  VkCommandBuffer BuildCommandBufferSimulation();
  void SetupPointsPipeline();
  void CleanupPipelineAndSwapchain();
  void RecreateSwapChain();
  void CreateUniformBuffer();
  void UpdateUniformBuffer(float a_time);

  void Cleanup();

  void SetupDeviceFeatures();
  void SetupDeviceExtensions();
  void SetupValidationLayers();
};


#endif //CHIMERA_SIMPLE_RENDER_H

