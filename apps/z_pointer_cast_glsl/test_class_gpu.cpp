#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_program.h"
#include "vk_copy.h"
#include "vk_buffer.h"
#include "vk_compute_pipeline.h"

#include "vk_rt_utils.h"
#include "rt_funcs.h"


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
  

  // query features for shaderInt8
  //
  VkPhysicalDeviceBufferDeviceAddressFeatures enabledDeviceAddressFeatures = {};
  enabledDeviceAddressFeatures.sType               = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
  enabledDeviceAddressFeatures.bufferDeviceAddress = VK_TRUE;
  enabledDeviceAddressFeatures.pNext               = nullptr;

  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;
  features.pNext      = &enabledDeviceAddressFeatures;
  
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
  deviceExtensions.push_back("VK_KHR_buffer_device_address"); 

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

  VkDescriptorType dtypes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
  uint32_t     dtypesizes[1] = {10};
  auto pBindings = std::make_shared<vkfw::ProgramBindings>(device, dtypes, dtypesizes, 1, 1);
  
  std::unique_ptr<vkfw::ComputePipelineMaker> pMaker = std::make_unique<vkfw::ComputePipelineMaker>();

  struct TestClassData
  {
    float a,b,c;
  };

  std::vector<TestClassData> data(10);
  for(size_t i=0;i<data.size();i++)
  {
    TestClassData test;
    test.a  = float(i);
    test.b  = float(i*2);
    test.c  = float(i*5);
    data[i] = test;
  }

  // (3) Create buffer
  //
  const size_t bufferSize1 = 10*sizeof(TestClassData);
  const size_t bufferSize2 = 10*sizeof(float);
  VkBuffer colorBuffer1    = vkfw::CreateBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer colorBuffer2    = vkfw::CreateBuffer(device, bufferSize2,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  
  VkDeviceMemory colorMem  = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {colorBuffer1, colorBuffer2});
  
  pCopyHelper->UpdateBuffer(colorBuffer1, 0, data.data(), data.size()*sizeof(TestClassData));

  VkDescriptorSet ds        = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE; 
  pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  pBindings->BindBuffer(0, colorBuffer1);
  pBindings->BindBuffer(1, colorBuffer2);
  pBindings->BindEnd(&ds, &dsl);
  
               pMaker->CreateShader(device, "shaders/pcast.comp.spv", nullptr, "main");
  auto pipeL = pMaker->MakeLayout(device, dsl, sizeof(uint64_t));
  auto pipeO = pMaker->MakePipeline(device);
  
  uint64_t zeroValue = vk_rt_utils::getBufferDeviceAddress(device, colorBuffer1);

  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool, 1)[0];
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    
    //vkCmdFillBuffer(commandBuffer, colorBuffer2, 0, VK_WHOLE_SIZE, 0);        // clear accumulated color
    
    vkCmdPushConstants     (commandBuffer, pipeL, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint64_t), &zeroValue);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeL, 0, 1, &ds, 0, nullptr);
    
    vkCmdBindPipeline      (commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeO);
    vkCmdDispatch          (commandBuffer, 1, 1, 1);

    vkEndCommandBuffer(commandBuffer);  

    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    float ms  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for full command buffer execution " << std::endl;

    std::vector<float> pixelData(10);
    pCopyHelper->ReadBuffer(colorBuffer2, 0, pixelData.data(), pixelData.size()*sizeof(uint32_t));

    std::cout << pixelData[0] << std::endl;
    std::cout << pixelData[1] << std::endl;
    std::cout << pixelData[2] << std::endl;
  }
  
  pMaker      = nullptr;
  pBindings   = nullptr;
  pCopyHelper = nullptr;

  vkDestroyBuffer(device, colorBuffer1, nullptr);
  vkDestroyBuffer(device, colorBuffer2, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}