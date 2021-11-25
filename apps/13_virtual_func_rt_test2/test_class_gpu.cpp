#include "include/BasicLogic.h" 
#include "Bitmap.h"

#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_descriptor_sets.h"
#include "vk_copy.h"
#include "vk_buffers.h"

#include "ray_tracing/vk_rt_utils.h"
//#include "ray_tracing/vk_rt_funcs.h"

#include "scene_mgr.h"

#include "vulkan_basics.h"
#include "test_class_generated.h"

using LiteMath::uint4;

class TestClass_GPU : public TestClass_Generated
{

public:
  TestClass_GPU(std::shared_ptr<SceneManager> a_pMgr, int a_maxThreads) : TestClass_Generated(a_maxThreads), m_pScnMgr(a_pMgr) 
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
  std::shared_ptr<vk_utils::DescriptorMaker> m_pBindings = nullptr;
  VkPipelineLayout      m_rtPipelineLayout = VK_NULL_HANDLE; 
  VkPipeline            m_rtPipeline       = VK_NULL_HANDLE; 

  bool m_enableHWAccel;

  void SetupRTPipeline(VkDevice a_device)
  {
    // first DS is from generated code, second is ours
    //
    std::vector<std::pair<VkDescriptorType, uint32_t> > dtypes = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,             1}
    };

    m_pBindings = std::make_shared<vk_utils::DescriptorMaker>(a_device, dtypes, 2);
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
    auto shaderCode = vk_utils::readSPVFile("shaders/raytrace.comp.spv");
    if(shaderCode.size() == 0)
      RUN_TIME_ERROR("[TestClass_GPU::SetupRTPipeline]: can not load shaders");
    VkShaderModule shaderModule = vk_utils::createShaderModule(a_device, shaderCode);
    
    VkComputePipelineCreateInfo pipelineCreateInfo = { VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineCreateInfo.stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName  = "main";
    pipelineCreateInfo.layout       = m_rtPipelineLayout;
    
    VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &m_rtPipeline));

    vkDestroyShaderModule(a_device, shaderModule, nullptr);
  }
  
  void RayTraceCmd(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar, Lite_Hit* out_hit, float2* out_bars) override
  {
    if(!m_enableHWAccel)
    {
      TestClass_Generated::RayTraceCmd(tid, rayPosAndNear, rayDirAndFar, out_hit, out_bars);
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
    pcData.m_sizeZ  = 1;
    pcData.m_tFlags = m_currThreadFlags;
    
    vkCmdPushConstants(m_currCmdBuffer, m_rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    
    VkDescriptorSet dsets[2] = {m_allGeneratedDS[1], m_rtDS};
    vkCmdBindDescriptorSets(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 0, 2, dsets, 0, nullptr);

    vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipeline);
    vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (pcData.m_sizeZ + blockSizeZ - 1) / blockSizeZ);
  
    VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
  }

  int LoadScene(const char* bvhPath, const char* meshPath, bool a_needReorder) override
  {
    if(TestClass_Generated::LoadScene(bvhPath, meshPath, !m_enableHWAccel) != 0 ) // may not load bvh actually!
      return 1; 

    // make scene from single mesh
    //  
    if(m_enableHWAccel)
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

void test_class_gpu()
{
  // (1) init vulkan
  //
  VkInstance       instance       = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkDevice         device         = VK_NULL_HANDLE;
  VkCommandPool    commandPool    = VK_NULL_HANDLE; 
  
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  std::vector<const char*> enabledLayers;
  std::vector<const char*> extensions;
  enabledLayers.push_back("VK_LAYER_KHRONOS_validation");
  enabledLayers.push_back("VK_LAYER_LUNARG_monitor");
  
  VK_CHECK_RESULT(volkInitialize());
  instance = vk_utils::createInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::findPhysicalDevice(instance, true, 0);
  auto queueComputeFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  // query features for RTX
  //
  //RTXDeviceFeatures rtxFeatures = SetupRTXFeatures(physicalDevice);

  // query features for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;
  features.pNext      = nullptr; //&rtxFeatures.m_enabledAccelStructFeatures;
  
  //VkPhysicalDeviceFeatures2 physDevFeatures2 = {};
  //physDevFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  //physDevFeatures2.pNext = &features;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  vk_utils::QueueFID_T fIDs = {};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;
  
  // Required by clspv for some reason
  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 
  
  // Required by VK_KHR_RAY_QUERY
  //deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
  //deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
  //
  //// Required by VK_KHR_acceleration_structure
  //deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  //deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  //deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

  // // Required by VK_KHR_ray_tracing_pipeline
  // m_deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
  // // Required by VK_KHR_spirv_1_4
  // m_deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

  fIDs.compute = queueComputeFID;
  device       = vk_utils::createLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures, 
                                               fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT, &features);
  volkLoadDevice(device);
  //vk_rt_utils::LoadRayTracingFunctions(device);

  commandPool  = vk_utils::createCommandPool(device, queueComputeFID, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT); 

  // (2) initialize vulkan helpers
  //  
  VkQueue computeQueue, transferQueue;
  {
    auto queueComputeFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue(device, queueComputeFID, 0, &computeQueue);
    vkGetDeviceQueue(device, queueComputeFID, 0, &transferQueue);
  }

  auto pCopyHelper = std::make_shared<vk_utils::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 8*1024*1024);
  auto pScnMgr     = std::make_shared<SceneManager>(device, physicalDevice, queueComputeFID, queueComputeFID, false); // todo: pass pCopyHelper
  auto pGPUImpl    = std::make_shared<TestClass_GPU>(pScnMgr, WIN_WIDTH*WIN_HEIGHT);  // !!! USING GENERATED CODE !!! 
  
  pGPUImpl->InitVulkanObjects(device, physicalDevice, WIN_WIDTH*WIN_HEIGHT); // !!! USING GENERATED CODE !!!                        
  pGPUImpl->LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.bvh", "../10_virtual_func_rt_test1/cornell_collapsed.vsgf", true);

  // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  //
  pGPUImpl->InitRandomGens(WIN_WIDTH*WIN_HEIGHT);                            // !!! USING GENERATED CODE !!!
  pGPUImpl->InitMemberBuffers();                                             // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSize1 = WIN_WIDTH*WIN_HEIGHT*sizeof(uint32_t);
  const size_t bufferSize2 = WIN_WIDTH*WIN_HEIGHT*sizeof(float)*4;
  VkBuffer xyBuffer        = vk_utils::createBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer colorBuffer1    = vk_utils::createBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer colorBuffer2    = vk_utils::createBuffer(device, bufferSize2,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  
  VkDeviceMemory colorMem  = vk_utils::allocateAndBindWithPadding(device, physicalDevice, {xyBuffer, colorBuffer1, colorBuffer2});
  
  pGPUImpl->SetVulkanInOutFor_PackXY(xyBuffer, 0);            // !!! USING GENERATED CODE !!! 

  pGPUImpl->SetVulkanInOutFor_CastSingleRay(xyBuffer,     0,  // !!! USING GENERATED CODE !!!
                                            colorBuffer1, 0); // !!! USING GENERATED CODE !!!

  pGPUImpl->SetVulkanInOutFor_NaivePathTrace(xyBuffer,   0,   // !!! USING GENERATED CODE !!!
                                             colorBuffer2,0); // !!! USING GENERATED CODE !!!

  //pGPUImpl->SetupRTPipeline(device);                          // !!! WRITE BY HAND        !!!
  pGPUImpl->UpdateAll(pCopyHelper);                           // !!! USING GENERATED CODE !!!
  
  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffers(device, commandPool, 1)[0];
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    //vkCmdFillBuffer(commandBuffer, xyBuffer, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->PackXYCmd(commandBuffer, WIN_WIDTH, WIN_HEIGHT, nullptr);       // !!! USING GENERATED CODE !!! 
    vkCmdFillBuffer(commandBuffer, colorBuffer2, 0, VK_WHOLE_SIZE, 0);        // clear accumulated color
    vkEndCommandBuffer(commandBuffer);  
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);

    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    pGPUImpl->CastSingleRayCmd(commandBuffer, WIN_WIDTH*WIN_HEIGHT, nullptr, nullptr);  // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
   
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    float ms  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for full command buffer execution " << std::endl;

    std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
    pCopyHelper->ReadBuffer(colorBuffer1, 0, pixelData.data(), pixelData.size()*sizeof(uint32_t));
    SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
   
    //return;
    std::cout << "begin path tracing passes ... " << std::endl;
    
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    pGPUImpl->NaivePathTraceCmd(commandBuffer, WIN_WIDTH*WIN_HEIGHT, 6, nullptr, nullptr);  // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
    
    start = std::chrono::high_resolution_clock::now();
    const int NUM_PASSES = 1000.0f;
    for(int i=0;i<NUM_PASSES;i++)
    {
      vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
      if(i % 100 == 0)
      {
        std::cout << "progress (gpu) = " << 100.0f*float(i)/float(NUM_PASSES) << "% \r";
        std::cout.flush();
      }
    }
    std::cout << std::endl;
    stop = std::chrono::high_resolution_clock::now();
    ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for " << NUM_PASSES << " times of command buffer execution " << std::endl;

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
  pGPUImpl    = nullptr;     
  pScnMgr     = nullptr;
  pCopyHelper = nullptr;

  vkDestroyBuffer(device, xyBuffer, nullptr);
  vkDestroyBuffer(device, colorBuffer1, nullptr);
  vkDestroyBuffer(device, colorBuffer2, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}