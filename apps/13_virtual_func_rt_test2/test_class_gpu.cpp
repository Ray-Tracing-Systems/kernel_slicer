#include "include/BasicLogic.h" 
#include "Bitmap.h"

#include <vector>
#include <iostream>
#include <memory>
#include <mutex>
#include <chrono>
#include <thread>

#include "vk_utils.h"
#include "vk_program.h"
#include "vk_copy.h"
#include "vk_buffer.h"

#include "vk_rt_utils.h"
#include "rt_funcs.h"

#include "scene_mgr.h"
#include "hello_tri.h"

#include "vulkan_basics.h"
#include "test_class_generated.h"

#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"

#ifdef WIN32
#pragma comment(lib,"glfw3.lib")
#endif

#define TILE_SIZE 256
//#define SINGLE       // don't split into tiles
#define BATCH_SUBMIT // submit all path tracing cmds at once
constexpr int NUM_PASSES  = 8;
constexpr int TOTAL_ITERS = 10;

using LiteMath::uint4;

class TestClass_GPU : public TestClass_Generated
{

public:
  TestClass_GPU(std::shared_ptr<SceneManager> a_pMgr) : m_pScnMgr(a_pMgr) 
  {
    m_enableHWAccel = false;
  }

  ~TestClass_GPU()
  {
    if(m_rtPipelineLayout) vkDestroyPipelineLayout(device, m_rtPipelineLayout, nullptr);
    if(m_rtPipeline)       vkDestroyPipeline      (device, m_rtPipeline,       nullptr);
  }
  
  VkDescriptorSet       m_rtDS       = nullptr;
  VkDescriptorSetLayout m_rtDSLayout = nullptr;
  std::shared_ptr<vkfw::ProgramBindings> m_pBindings = nullptr;
  VkPipelineLayout      m_rtPipelineLayout = VK_NULL_HANDLE; 
  VkPipeline            m_rtPipeline       = VK_NULL_HANDLE; 

  bool m_enableHWAccel;

  void SetupRTPipeline(VkDevice a_device)
  {
    // first DS is from generated code, second is ours
    //
    VkDescriptorType dtypes[2] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    uint32_t     dtypesizes[2] = {1, 1};
    m_pBindings = std::make_shared<vkfw::ProgramBindings>(a_device, dtypes, dtypesizes, 2, 1);
    
    m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
    m_pBindings->BindAccelStruct(0, m_pScnMgr->getTLAS().handle);
    //m_pBindings->BindBuffer     (1, CastSingleRay_local.out_colorBuffer, CastSingleRay_local.out_colorOffset);
    m_pBindings->BindEnd(&m_rtDS, &m_rtDSLayout);
    
    VkDescriptorSetLayout inputSets[2] = {RayTraceDSLayout , m_rtDSLayout};

    VkPushConstantRange  pcRange;
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = 4*sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges    = &pcRange;
    pipelineLayoutInfo.pSetLayouts            = inputSets;
    pipelineLayoutInfo.setLayoutCount         = 2;
    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutInfo, nullptr, &m_rtPipelineLayout));
    
    // load shader and create compute pipeline for RTX accelerated ray tracing via GLSL
    //
    auto shaderCode = vk_utils::ReadFile("shaders/raytrace.comp.spv");
    if(shaderCode.size() == 0)
      RUN_TIME_ERROR("[TestClass_GPU::SetupRTPipeline]: can not load shaders");
    VkShaderModule shaderModule = vk_utils::CreateShaderModule(a_device, shaderCode);
    
    VkComputePipelineCreateInfo pipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName  = "main";
    pipelineCreateInfo.layout       = m_rtPipelineLayout;
    
    VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &m_rtPipeline));

    vkDestroyShaderModule(a_device, shaderModule, nullptr);
  }
  
  void RayTraceCmd(uint32_t tid, const float4* rayPosAndNear, float4* rayDirAndFar, Lite_Hit* out_hit, float2* out_bars, uint32_t tileOffset) override
  {
    if(!m_enableHWAccel)
    {
      TestClass_Generated::RayTraceCmd(tid, rayPosAndNear, rayDirAndFar, out_hit, out_bars, tileOffset);
      return;
    }
    uint32_t blockSizeX = 256;
    uint32_t blockSizeY = 1;
    uint32_t blockSizeZ = 1;
  
    struct KernelArgsPC
    {
      uint32_t m_sizeX;
      uint32_t m_sizeY;
      uint32_t m_sizeZ;
      uint32_t m_tFlags;
    } pcData;
    
    pcData.m_sizeX  = tid;
    pcData.m_sizeY  = 1;
    pcData.m_sizeZ  = tileOffset;
    pcData.m_tFlags = m_currThreadFlags;
    
    vkCmdPushConstants(m_currCmdBuffer, m_rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    
    VkDescriptorSet dsets[2] = {m_allGeneratedDS[1], m_rtDS};
    vkCmdBindDescriptorSets(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 0, 2, dsets, 0, nullptr);

    vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipeline);
    vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (1 + blockSizeZ - 1) / blockSizeZ);
  
    VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);
  }

  int LoadScene(const char* bvhPath, const char* meshPath, bool a_needReorder, bool a_initMgr) override
  {
    if(TestClass_Generated::LoadScene(bvhPath, meshPath, !m_enableHWAccel, a_initMgr) != 0 ) // may not load bvh actually!
      return 1; 

    // make scene from single mesh
    //
    if(a_initMgr)
    {
      m_pScnMgr->LoadSingleMesh(meshPath);
      m_pScnMgr->BuildAllBLAS();
      m_pScnMgr->BuildTLAS();
    }
    
    return 0;
  }

  std::shared_ptr<SceneManager> m_pScnMgr;
};

struct RTXDeviceFeatures
{
  VkPhysicalDeviceAccelerationStructureFeaturesKHR m_enabledAccelStructFeatures{};
  VkPhysicalDeviceBufferDeviceAddressFeatures      m_enabledDeviceAddressFeatures{};
  VkPhysicalDeviceRayQueryFeaturesKHR              m_enabledRayQueryFeatures;
};

static RTXDeviceFeatures SetupRTXFeatures(VkPhysicalDevice a_physDev)
{
  static RTXDeviceFeatures g_rtFeatures;

  g_rtFeatures.m_enabledRayQueryFeatures.sType    = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
  g_rtFeatures.m_enabledRayQueryFeatures.rayQuery = VK_TRUE;

  g_rtFeatures.m_enabledDeviceAddressFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  g_rtFeatures.m_enabledDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
  g_rtFeatures.m_enabledDeviceAddressFeatures.pNext               = &g_rtFeatures.m_enabledRayQueryFeatures;

  g_rtFeatures.m_enabledAccelStructFeatures.sType                 = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
  g_rtFeatures.m_enabledAccelStructFeatures.accelerationStructure = VK_TRUE;
  g_rtFeatures.m_enabledAccelStructFeatures.pNext                 = &g_rtFeatures.m_enabledDeviceAddressFeatures;

  return g_rtFeatures;
}

void test_class_gpu_V2()
{
  auto pWindowDummy = CreateHelloTriImpl();
  pWindowDummy->InitWindow();

  // (1) init vulkan
  //
  VkInstance       instance       = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;

#ifndef NDEBUG
  bool enableValidationLayers = true;
#else
  bool enableValidationLayers = false;
#endif

  std::vector<const char*> enabledLayers;
  std::vector<const char*> extensions;
  enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
  enabledLayers.push_back("VK_LAYER_LUNARG_monitor");
  
  // add glfw to create window
  {
    uint32_t glfwExtensionCount = 0;
    const char** glfwExtensions;
    glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    auto ext2      = std::vector<const char*>(glfwExtensions, glfwExtensions + glfwExtensionCount);
    extensions.insert(extensions.end(), ext2.begin(), ext2.end());
  }

  VK_CHECK_RESULT(volkInitialize());
  instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::FindPhysicalDevice(instance, true, 1);
  auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT);
  auto queueComputeFID2 = vk_utils::GetDifferentQueueFamilyIndex(physicalDevice, VK_QUEUE_COMPUTE_BIT, queueComputeFID);
  auto queueTransferFID = 2u;//vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT);
  // query features for RTX
  //
  //RTXDeviceFeatures rtxFeatures = SetupRTXFeatures(physicalDevice);

  // query features for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;
  features.pNext      = nullptr;//&rtxFeatures.m_enabledAccelStructFeatures;

  VkPhysicalDeviceFeatures2 physDevFeatures2 = {};
  physDevFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  physDevFeatures2.pNext = &features;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  vk_utils::queueFamilyIndices fIDs = {.graphics = queueComputeFID, .compute = queueComputeFID2, .transfer = queueTransferFID};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;

  // Required by clspv for some reason
  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8");
  deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

  // Required by VK_KHR_RAY_QUERY
 /* deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);

  // Required by VK_KHR_acceleration_structure
  deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);*/

  // // Required by VK_KHR_ray_tracing_pipeline
  // m_deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
  // // Required by VK_KHR_spirv_1_4
  // m_deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

  constexpr uint32_t nComputeQs = 2;

  device       = vk_utils::CreateLogicalDevice2(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures,
                                                fIDs, nComputeQs,
                                                VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT,
                                                physDevFeatures2);
  volkLoadDevice(device);
  //LoadRayTracingFunctions(device);

  auto commandPool1  = vk_utils::CreateCommandPool2(device, physicalDevice, queueComputeFID, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
  auto commandPool2  = vk_utils::CreateCommandPool2(device, physicalDevice, queueComputeFID2, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // (2) initialize vulkan helpers
  //
  VkQueue transferQueue;
  std::array<VkQueue, nComputeQs> computeQueues = {VK_NULL_HANDLE};
  {
    vkGetDeviceQueue(device, queueTransferFID, 0, &transferQueue);
    vkGetDeviceQueue(device, queueComputeFID, 0, &computeQueues[0]);
    vkGetDeviceQueue(device, queueComputeFID2, 0, &computeQueues[1]);
  }

  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueTransferFID, 8*1024*1024);
  auto pScnMgr     = std::make_shared<SceneManager>(device, physicalDevice, queueTransferFID, queueComputeFID, pCopyHelper, false);

  auto pGPUImpl1    = std::make_shared<TestClass_GPU>(pScnMgr);
  auto pGPUImpl2    = std::make_shared<TestClass_GPU>(pScnMgr);

  constexpr uint32_t totalWork = WIN_WIDTH*WIN_HEIGHT;

#ifdef SINGLE
  constexpr uint32_t nTiles = 1;
  constexpr uint32_t perTile = totalWork;
#else
  constexpr uint32_t perTile = TILE_SIZE * TILE_SIZE;
  constexpr uint32_t nTiles = totalWork / perTile;
#endif


  pGPUImpl1->InitVulkanObjects(device, physicalDevice, perTile);
  pGPUImpl2->InitVulkanObjects(device, physicalDevice, perTile);

  pGPUImpl1->LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.bvh", "../10_virtual_func_rt_test1/cornell_collapsed.vsgf",
                       true, false);
  pGPUImpl2->LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.bvh", "../10_virtual_func_rt_test1/cornell_collapsed.vsgf",
                       true, false);

  // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  //
  pGPUImpl1->InitRandomGens(perTile);
  pGPUImpl1->InitMemberBuffers();

  pGPUImpl2->InitRandomGens(perTile);
  pGPUImpl2->InitMemberBuffers();

  // (3) Create buffer
  //
  const size_t bufferSize1 = WIN_WIDTH*WIN_HEIGHT*sizeof(uint32_t);
  const size_t bufferSize2 = WIN_WIDTH*WIN_HEIGHT*sizeof(float)*4;
#ifdef SINGLE
  VkBuffer xyBuffer        = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                VK_SHARING_MODE_EXCLUSIVE);
  VkBuffer colorBuffer1    = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                VK_SHARING_MODE_EXCLUSIVE);
  VkBuffer colorBuffer2    = vkfw::CreateBuffer(device, bufferSize2,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                VK_SHARING_MODE_EXCLUSIVE);
#else
  VkBuffer xyBuffer        = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                VK_SHARING_MODE_CONCURRENT);
  VkBuffer colorBuffer1    = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                VK_SHARING_MODE_CONCURRENT);
  VkBuffer colorBuffer2    = vkfw::CreateBuffer(device, bufferSize2,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                                VK_SHARING_MODE_CONCURRENT);
#endif
  VkDeviceMemory colorMem  = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {xyBuffer, colorBuffer1, colorBuffer2});

  pGPUImpl1->SetVulkanInOutFor_PackXY(xyBuffer, 0);            // !!! USING GENERATED CODE !!!
  pGPUImpl1->SetVulkanInOutFor_CastSingleRay(xyBuffer,     0,  // !!! USING GENERATED CODE !!!
                                            colorBuffer1, 0); // !!! USING GENERATED CODE !!!
  pGPUImpl1->SetVulkanInOutFor_NaivePathTrace(xyBuffer,   0,   // !!! USING GENERATED CODE !!!
                                             colorBuffer2,0); // !!! USING GENERATED CODE !!!
  //pGPUImpl1->SetupRTPipeline(device);                          // !!! WRITE BY HAND        !!!
  pGPUImpl1->UpdateAll(pCopyHelper);                           // !!! USING GENERATED CODE !!!

  // ***

  pGPUImpl2->SetVulkanInOutFor_PackXY(xyBuffer, 0);           // !!! USING GENERATED CODE !!!
  pGPUImpl2->SetVulkanInOutFor_CastSingleRay(xyBuffer,     0, // !!! USING GENERATED CODE !!!
                                            colorBuffer1, 0); // !!! USING GENERATED CODE !!!
  pGPUImpl2->SetVulkanInOutFor_NaivePathTrace(xyBuffer,   0,  // !!! USING GENERATED CODE !!!
                                             colorBuffer2,0); // !!! USING GENERATED CODE !!!
  //pGPUImpl2->SetupRTPipeline(device);                       // !!! WRITE BY HAND        !!!
  pGPUImpl2->UpdateAll(pCopyHelper);                          // !!! USING GENERATED CODE !!!

  pWindowDummy->Init(instance, physicalDevice, device);
  
  int totalRuns = 0;

  // now compute some thing useful
  //
  {
    {
      VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool1, 1)[0];
      VkCommandBufferBeginInfo beginCommandBufferInfo = {};
      beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
      //vkCmdFillBuffer(commandBuffer, xyBuffer, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
      pGPUImpl1->PackXYCmd(commandBuffer, WIN_WIDTH, WIN_HEIGHT, nullptr);       // !!! USING GENERATED CODE !!!
      vkCmdFillBuffer(commandBuffer, colorBuffer2, 0, VK_WHOLE_SIZE, 0);        // clear accumulated color
      vkEndCommandBuffer(commandBuffer);
      vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueues[0], device);
      pWindowDummy->DoFrame();
    }
#ifndef SINGLE
    {
      VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool2, 1)[0];
      VkCommandBufferBeginInfo beginCommandBufferInfo = {};
      beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
      //vkCmdFillBuffer(commandBuffer, xyBuffer, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
      pGPUImpl2->PackXYCmd(commandBuffer, WIN_WIDTH, WIN_HEIGHT, nullptr);       // !!! USING GENERATED CODE !!!
      vkCmdFillBuffer(commandBuffer, colorBuffer2, 0, VK_WHOLE_SIZE, 0);        // clear accumulated color
      vkEndCommandBuffer(commandBuffer);
      vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueues[1], device);
      pWindowDummy->DoFrame();
    }
#endif

    uint32_t tileStart = 0;
    uint32_t tileEnd   = tileStart + perTile;

#ifdef SINGLE
    std::vector<VkCommandBuffer> singleRayCmds1 = vk_utils::CreateCommandBuffers(device, commandPool1, 1);
#else
    std::vector<VkCommandBuffer> singleRayCmds1 = vk_utils::CreateCommandBuffers(device, commandPool1, nTiles / 2);
    std::vector<VkCommandBuffer> singleRayCmds2 = vk_utils::CreateCommandBuffers(device, commandPool2, nTiles / 2);
#endif
    VkCommandBufferBeginInfo tileBeginCmdInfo = {};
    tileBeginCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    tileBeginCmdInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    // ***** Record *****
    uint32_t idx = 0;
    for(auto j = 0; j < nTiles; ++j)
    {
      tileStart = perTile * j;
      tileEnd   = tileStart + perTile;

#ifndef SINGLE
      if(j % 2 == 0)
#endif
      {
        vkBeginCommandBuffer(singleRayCmds1[idx], &tileBeginCmdInfo);
        pGPUImpl1->CastSingleRayCmd(singleRayCmds1[idx], totalWork, nullptr, nullptr, tileStart,
                                    tileEnd);  // !!! USING GENERATED CODE !!!
        vkEndCommandBuffer(singleRayCmds1[idx]);
      }
#ifndef SINGLE
      else
      {
        vkBeginCommandBuffer(singleRayCmds2[idx], &tileBeginCmdInfo);
        pGPUImpl2->CastSingleRayCmd(singleRayCmds2[idx], totalWork, nullptr, nullptr, tileStart,
                                    tileEnd);  // !!! USING GENERATED CODE !!!
        vkEndCommandBuffer(singleRayCmds2[idx]);

        ++idx;
      }
#endif
    }
    
 
    // ***** Execute *****
    {
      auto start = std::chrono::high_resolution_clock::now();
      //assert(singleRayCmds1.size() == singleRayCmds2.size());
      for (auto idx = 0; idx < singleRayCmds1.size(); ++idx)
      {
        VkFence fence1 = vk_utils::SubmitCommandBuffer(singleRayCmds1[idx], computeQueues[0], device);
#ifndef SINGLE
        VkFence fence2 = vk_utils::SubmitCommandBuffer(singleRayCmds2[idx], computeQueues[1], device);
#endif

        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence1, VK_TRUE, vk_utils::FENCE_TIMEOUT));
        pWindowDummy->DoFrame();
        vkDestroyFence(device, fence1, NULL);

#ifndef SINGLE
        VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence2, VK_TRUE, vk_utils::FENCE_TIMEOUT));
        vkDestroyFence(device, fence2, NULL);
#endif

      }
      auto stop = std::chrono::high_resolution_clock::now();
      float ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
      std::cout << "CastSingleRay, all tiles: " << ms << " ms for full command buffer execution " << std::endl;
    }

    totalRuns = 0;
    BEGIN_PATH_TRACING_AGAIN: 

    tileStart = 0;
    tileEnd   = tileStart + perTile;

    std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
    pCopyHelper->ReadBuffer(colorBuffer1, 0, pixelData.data(), pixelData.size()*sizeof(uint32_t));
    SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

    std::cout << "begin path tracing passes ... " << std::endl;

#ifdef SINGLE
    std::vector<VkCommandBuffer> pathCmds1 = vk_utils::CreateCommandBuffers(device, commandPool1, 1);
#else
    std::vector<VkCommandBuffer> pathCmds1 = vk_utils::CreateCommandBuffers(device, commandPool1, nTiles / 2);
    std::vector<VkCommandBuffer> pathCmds2 = vk_utils::CreateCommandBuffers(device, commandPool2, nTiles / 2);
#endif

    VkCommandBufferBeginInfo pathBeginCmdInfo = {};
    pathBeginCmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    pathBeginCmdInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

    // ***** Record *****
    idx = 0;
    for(auto j = 0; j < nTiles; ++j)
    {
      tileStart = perTile * j;
      tileEnd   = tileStart + perTile;

#ifndef SINGLE
      if(j % 2 == 0)
#endif
      {
        vkBeginCommandBuffer(pathCmds1[idx], &pathBeginCmdInfo);
        pGPUImpl1->NaivePathTraceCmd(pathCmds1[idx], totalWork, 6, nullptr, nullptr, tileStart,
                                     tileEnd);  // !!! USING GENERATED CODE !!!
        vkEndCommandBuffer(pathCmds1[idx]);
      }
#ifndef SINGLE
      else
      {
        vkBeginCommandBuffer(pathCmds2[idx], &pathBeginCmdInfo);
        pGPUImpl2->NaivePathTraceCmd(pathCmds2[idx], totalWork, 6, nullptr, nullptr, tileStart,
                                     tileEnd);  // !!! USING GENERATED CODE !!!
        vkEndCommandBuffer(pathCmds2[idx]);
        ++idx;
      }
#endif
    }

    // ***** Execute (multithreaded submission)*****
    {
      std::vector<std::thread> workers(nComputeQs);

      auto start = std::chrono::high_resolution_clock::now();

      auto work = [device,pWindowDummy](VkQueue q, std::vector<VkCommandBuffer> &cmds){
          VkFence fence;
          VkFenceCreateInfo fenceCreateInfo = {};
          fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
          fenceCreateInfo.flags = 0;
          VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, NULL, &fence));
#ifdef BATCH_SUBMIT
          std::vector<VkSubmitInfo> submits(NUM_PASSES);
          for (int i = 0; i < NUM_PASSES; i++)
          {
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = cmds.size();
            submitInfo.pCommandBuffers = cmds.data();
            submits[i] = submitInfo;
          }

          VK_CHECK_RESULT(vkQueueSubmit(q, submits.size(), submits.data(), fence));
          VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, vk_utils::FENCE_TIMEOUT));
          pWindowDummy->DoFrame();
          VK_CHECK_RESULT(vkResetFences(device, 1, &fence));
#else
          for (int i = 0; i < NUM_PASSES; i++)
          {
            VkSubmitInfo submitInfo = {};
            submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
            submitInfo.commandBufferCount = cmds.size();
            submitInfo.pCommandBuffers = cmds.data();

            VK_CHECK_RESULT(vkQueueSubmit(q, 1, &submitInfo, fence));
            VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, vk_utils::FENCE_TIMEOUT));
            pWindowDummy->DoFrame();
            VK_CHECK_RESULT(vkResetFences(device, 1, &fence));
          }
#endif
          vkDestroyFence(device, fence, NULL);
      };

      workers[0] = std::move(std::thread(work, computeQueues[0], std::ref(pathCmds1)));
#ifndef SINGLE
      workers[1] = std::move(std::thread(work, computeQueues[1], std::ref(pathCmds2)));
#endif

      for (auto j = 0; j < workers.size(); ++j)
      {
        if(workers[j].joinable())
          workers[j].join();
      }
      auto stop = std::chrono::high_resolution_clock::now();
      float ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
      std::cout << "Path tracing, all tiles: " << ms << " ms for " << NUM_PASSES
                << " times of command buffer execution " << std::endl;
    }

    // ***** Execute *****
//    {
//      std::vector<VkFence> fences(2);
//      auto start = std::chrono::high_resolution_clock::now();
//      for (int i = 0; i < NUM_PASSES; i++)
//      {
//        fences[0] = vk_utils::SubmitCommandBuffers(pathCmds1, computeQueues[0], device);
//        fences[1] = vk_utils::SubmitCommandBuffers(pathCmds2, computeQueues[1], device);
//
//        vkWaitForFences(device, 2, fences.data(), VK_TRUE, vk_utils::FENCE_TIMEOUT);
//        for (uint j = 0; j < 2; ++j)
//        {
////          VK_CHECK_RESULT(vkWaitForFences(device, 1, &fences[j], VK_TRUE, vk_utils::FENCE_TIMEOUT));
//          vkDestroyFence(device, fences[j], NULL);
//        }
//        if (i % 100 == 0)
//        {
//          std::cout << "progress (gpu) = " << 100.0f * float(i) / float(NUM_PASSES) << "% \r";
//          std::cout.flush();
//        }
//      }
//      std::cout << std::endl;
//      auto stop = std::chrono::high_resolution_clock::now();
//      float ms = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.f;
//      std::cout << "Path tracing, all tiles: " << ms << " ms for " << NUM_PASSES
//                << " times of command buffer execution " << std::endl;
//
//    }

    if(totalRuns < TOTAL_ITERS)
    {
      totalRuns++;
      goto BEGIN_PATH_TRACING_AGAIN;
    }

    std::vector<float4> pixelsf(WIN_WIDTH*WIN_HEIGHT);
    pCopyHelper->ReadBuffer(colorBuffer2, 0, pixelsf.data(), pixelsf.size()*sizeof(float4));

    const float normConst = 1.0f/float(NUM_PASSES);
    const float invGamma  = 1.0f / 2.2f;

    for(int i=0;i<WIN_HEIGHT*WIN_HEIGHT;i++)
    {
      float4 color = pixelsf[i]*normConst;
      color.x      = powf(color.x, invGamma);
      color.y      = powf(color.y, invGamma);
      color.z      = powf(color.z, invGamma);
      color.w      = 1.0f;
      pixelData[i] = RealColorToUint32(clamp(color, 0.0f, 1.0f));
    }
    SaveBMP("zout_gpu2.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
    std::cout << std::endl;

  }

  // (6) destroy and free resources before exit
  //
  pGPUImpl1    = nullptr;
  pGPUImpl2    = nullptr;

  pScnMgr      = nullptr;
  pCopyHelper  = nullptr;
  pWindowDummy = nullptr;

  vkDestroyBuffer(device, xyBuffer, nullptr);
  vkDestroyBuffer(device, colorBuffer1, nullptr);
  vkDestroyBuffer(device, colorBuffer2, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool1, nullptr);
  vkDestroyCommandPool(device, commandPool2, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}