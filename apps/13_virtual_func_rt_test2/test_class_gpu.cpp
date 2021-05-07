#include "include/BasicLogic.h" 
#include "Bitmap.h"

#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_program.h"
#include "vk_copy.h"
#include "vk_buffer.h"

#include "vk_rt_utils.h"
#include "rt_funcs.h"

#include "scene_mgr.h"

#include "vulkan_basics.h"
#include "test_class_generated.h"

using LiteMath::uint4;

class TestClass_GPU : public TestClass_Generated
{

public:
  TestClass_GPU(std::shared_ptr<SceneManager> a_pMgr) : m_pScnMgr(a_pMgr) 
  {

  }

  ~TestClass_GPU()
  {
    if(m_rtPipelineLayout) vkDestroyPipelineLayout(device, m_rtPipelineLayout, nullptr);
    if(m_rtPipeline)       vkDestroyPipeline      (device, m_rtPipeline,       nullptr);
  }

  void InitVulkanObjects(VkDevice a_device, VkPhysicalDevice a_physicalDevice, size_t a_maxThreadsCount) override
  {
    TestClass_Generated::InitVulkanObjects(a_device, a_physicalDevice, a_maxThreadsCount);
    SetupRTPipeline(a_device);
  }
  
  VkDescriptorSet       m_rtDS       = nullptr;
  VkDescriptorSetLayout m_rtDSLayout = nullptr;
  std::shared_ptr<vkfw::ProgramBindings> m_pBindings = nullptr;
  VkPipelineLayout      m_rtPipelineLayout = VK_NULL_HANDLE; 
  VkPipeline            m_rtPipeline       = VK_NULL_HANDLE; 

  void SetupRTPipeline(VkDevice a_device)
  {
    // first DS is from generated code, second is ours
    //
    VkDescriptorType dtypes[2] = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
    uint32_t     dtypesizes[2] = {1, 1};
    m_pBindings = std::make_shared<vkfw::ProgramBindings>(a_device, dtypes, dtypesizes, 2, 1);
    
    m_pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
    m_pBindings->BindAccelStruct(0, m_pScnMgr->getTLAS().handle);
    m_pBindings->BindBuffer     (1, CastSingleRay_local.out_colorBuffer, CastSingleRay_local.out_colorOffset);
    m_pBindings->BindEnd(&m_rtDS, &m_rtDSLayout);
    
    VkDescriptorSetLayout inputSets[2] = {RayTraceDSLayout , m_rtDSLayout};

    VkPipelineLayoutCreateInfo pipelineLayoutInfo = {};
    pipelineLayoutInfo.sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayoutInfo.pPushConstantRanges    = nullptr;
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
  
  void RayTraceCmd(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar, Lite_Hit* out_hit, float2* out_bars) override
  {
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
  
    vkCmdPushConstants(m_currCmdBuffer, RayTraceLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(KernelArgsPC), &pcData);
    
    vkCmdBindPipeline(m_currCmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, RayTracePipeline);
    vkCmdDispatch    (m_currCmdBuffer, (pcData.m_sizeX + blockSizeX - 1) / blockSizeX, (pcData.m_sizeY + blockSizeY - 1) / blockSizeY, (pcData.m_sizeZ + blockSizeZ - 1) / blockSizeZ);
  
    VkMemoryBarrier memoryBarrier = { VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT };
    vkCmdPipelineBarrier(m_currCmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0, nullptr);  
  }

  int LoadScene(const char* bvhPath, const char* meshPath) override
  {
    if(TestClass_Generated::LoadScene(bvhPath, meshPath) != 0 ) // may not load bvh actually!
      return 1; 

    // make scene from single mesh
    //  
    m_pScnMgr->LoadSingleMesh(meshPath);
    m_pScnMgr->BuildAllBLAS();
    m_pScnMgr->BuildTLAS();
    
    return 0;
  }

  std::shared_ptr<SceneManager> m_pScnMgr;
};

struct RTXDeviceFeatures
{
  VkPhysicalDeviceAccelerationStructureFeaturesKHR m_accelStructFeatures{};
  VkPhysicalDeviceAccelerationStructureFeaturesKHR m_enabledAccelStructFeatures{};
  VkPhysicalDeviceBufferDeviceAddressFeatures      m_enabledDeviceAddressFeatures{};
  VkPhysicalDeviceRayQueryFeaturesKHR              m_enabledRayQueryFeatures;
};

static RTXDeviceFeatures SetupRTXFeatures(VkPhysicalDevice a_physDev)
{
  static RTXDeviceFeatures g_rtFeatures;

  g_rtFeatures.m_accelStructFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;

  VkPhysicalDeviceFeatures2 deviceFeatures2{};
  deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  deviceFeatures2.pNext = &g_rtFeatures.m_accelStructFeatures;
  vkGetPhysicalDeviceFeatures2(a_physDev, &deviceFeatures2);

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
  instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::FindPhysicalDevice(instance, true, 1);
  auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  // query features for RTX
  //
  RTXDeviceFeatures rtxFeatures = SetupRTXFeatures(physicalDevice);

  // query features for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;
  features.pNext      = &rtxFeatures.m_enabledAccelStructFeatures;
  
  VkPhysicalDeviceFeatures2 physDevFeatures2 = {};
  physDevFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  physDevFeatures2.pNext = &features;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  vk_utils::queueFamilyIndices fIDs = {};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;
  
  // Required by clspv for some reason
  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 
  
  // Required by VK_KHR_RAY_QUERY
  deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);

  // Required by VK_KHR_acceleration_structure
  deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
  deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
  deviceExtensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);

  // // Required by VK_KHR_ray_tracing_pipeline
  // m_deviceExtensions.push_back(VK_KHR_SPIRV_1_4_EXTENSION_NAME);
  // // Required by VK_KHR_spirv_1_4
  // m_deviceExtensions.push_back(VK_KHR_SHADER_FLOAT_CONTROLS_EXTENSION_NAME);

  fIDs.compute = queueComputeFID;
  device       = vk_utils::CreateLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures, 
                                               fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT, physDevFeatures2);
  volkLoadDevice(device);
  LoadRayTracingFunctions(device);

  commandPool  = vk_utils::CreateCommandPool(device, physicalDevice, VK_QUEUE_COMPUTE_BIT, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // (2) initialize vulkan helpers
  //  
  VkQueue computeQueue, transferQueue;
  {
    auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue(device, queueComputeFID, 0, &computeQueue);
    vkGetDeviceQueue(device, queueComputeFID, 0, &transferQueue);
  }


  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 8*1024*1024);\
  auto pScnMgr     = std::make_shared<SceneManager>(device, physicalDevice, queueComputeFID, queueComputeFID, pCopyHelper, true, true);
  auto pGPUImpl    = std::make_shared<TestClass_GPU>(pScnMgr);               // !!! USING GENERATED CODE !!! 
  
  pGPUImpl->InitVulkanObjects(device, physicalDevice, WIN_WIDTH*WIN_HEIGHT); // !!! USING GENERATED CODE !!!                        
  pGPUImpl->LoadScene("../10_virtual_func_rt_test1/cornell_collapsed.bvh", "../10_virtual_func_rt_test1/cornell_collapsed.vsgf");

  // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  //
  pGPUImpl->InitRandomGens(WIN_WIDTH*WIN_HEIGHT);                            // !!! USING GENERATED CODE !!!
  pGPUImpl->InitMemberBuffers();                                             // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSize1 = WIN_WIDTH*WIN_HEIGHT*sizeof(uint32_t);
  const size_t bufferSize2 = WIN_WIDTH*WIN_HEIGHT*sizeof(float)*4;
  VkBuffer xyBuffer        = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer colorBuffer1    = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer colorBuffer2    = vkfw::CreateBuffer(device, bufferSize2,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  
  VkDeviceMemory colorMem  = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {xyBuffer, colorBuffer1, colorBuffer2});
  
  pGPUImpl->SetVulkanInOutFor_PackXY(xyBuffer, 0);            // !!! USING GENERATED CODE !!! 

  pGPUImpl->SetVulkanInOutFor_CastSingleRay(xyBuffer,     0,  // !!! USING GENERATED CODE !!!
                                            colorBuffer1, 0); // !!! USING GENERATED CODE !!!

  pGPUImpl->SetVulkanInOutFor_NaivePathTrace(xyBuffer,   0,   // !!! USING GENERATED CODE !!!
                                             colorBuffer2,0); // !!! USING GENERATED CODE !!!

  pGPUImpl->UpdateAll(pCopyHelper);                           // !!! USING GENERATED CODE !!!
  
  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool, 1)[0];
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    //vkCmdFillBuffer(commandBuffer, xyBuffer, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->PackXYCmd(commandBuffer, WIN_WIDTH, WIN_HEIGHT, nullptr);       // !!! USING GENERATED CODE !!! 
    vkCmdFillBuffer(commandBuffer, colorBuffer2, 0, VK_WHOLE_SIZE, 0);        // clear accumulated color
    vkEndCommandBuffer(commandBuffer);  
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);

    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    pGPUImpl->CastSingleRayCmd(commandBuffer, WIN_WIDTH*WIN_HEIGHT, nullptr, nullptr);  // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
   
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    float ms  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for full command buffer execution " << std::endl;

    std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
    pCopyHelper->ReadBuffer(colorBuffer1, 0, pixelData.data(), pixelData.size()*sizeof(uint32_t));
    SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);
    
    return;

    //TestClass_UBO_Data testData;
    //pGPUImpl->ReadClassData(pCopyHelper, &testData);
    //int a = 2;

    // auto objPointers = pGPUImpl->GetObjPtrArray(pCopyHelper);
    // {
    //   uint currTag = -1, begOffs = 0;
    //   for(size_t i=0;i<objPointers.size();i++)
    //   {
    //     const uint kgen_objTag    = (objPointers[i].x & IMaterial::TAG_MASK) >> (32 - IMaterial::TAG_BITS);
    //     const uint kgen_objOffset = (objPointers[i].x & IMaterial::OFS_MASK);
    //     
    //     if(currTag != kgen_objTag)
    //     {
    //       if(currTag != -1)
    //         std::cout << currTag << ": " << begOffs << " : " << i << "]" << std::endl;
    //       begOffs = i;
    //       currTag = kgen_objTag;
    //     }
    //   }
    // }
    // int b = 3;

    //auto testIndirect = pGPUImpl->GetIndirectBufferData(pCopyHelper);
    //int c = 4;

    std::cout << "begin path tracing passes ... " << std::endl;
    
    vkResetCommandBuffer(commandBuffer, 0);
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    pGPUImpl->NaivePathTraceCmd(commandBuffer, WIN_WIDTH*WIN_HEIGHT, 6, nullptr, nullptr);  // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
    
    start = std::chrono::high_resolution_clock::now();
    const int NUM_PASSES = 1000.0f;
    for(int i=0;i<NUM_PASSES;i++)
    {
      vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
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
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!! 

  vkDestroyBuffer(device, xyBuffer, nullptr);
  vkDestroyBuffer(device, colorBuffer1, nullptr);
  vkDestroyBuffer(device, colorBuffer2, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}