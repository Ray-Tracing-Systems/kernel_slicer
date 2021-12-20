#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_descriptor_sets.h"
#include "vk_copy.h"
#include "vk_buffers.h"

#include "test_class_generated.h"

#include "include/ArrayProcess_ubo.h"

class ArrayProcess_GPU : public ArrayProcess_Generated
{
public:
  ArrayProcess_GPU(){}

  VkBufferUsageFlags GetAdditionalFlagsForUBO() const override { return VK_BUFFER_USAGE_TRANSFER_SRC_BIT; }
  VkBuffer GiveMeUBO() { return m_classDataBuffer; }
};

int32_t array_summ_gpu(const std::vector<int32_t>& inArrayCPU)
{
  int32_t resSumm = 0;
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
  enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
  
  VK_CHECK_RESULT(volkInitialize());
  instance = vk_utils::createInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::findPhysicalDevice(instance, true, 0);
  auto queueComputeFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  VkPhysicalDeviceVariablePointersFeatures varPointers = {};
  varPointers.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
  varPointers.pNext = nullptr;
  varPointers.variablePointers              = VK_TRUE;
  varPointers.variablePointersStorageBuffer = VK_TRUE;

  // query for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.pNext      = &varPointers;
  features.shaderInt8 = VK_TRUE;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;
  vk_utils::QueueFID_T fIDs = {};

  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 
  deviceExtensions.push_back(VK_KHR_VARIABLE_POINTERS_EXTENSION_NAME); // some validation layer says we need this 

  fIDs.compute = queueComputeFID;
  device       = vk_utils::createLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures,
                                               fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT, &features);
  volkLoadDevice(device);

  commandPool  = vk_utils::createCommandPool(device, fIDs.compute, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // (2) initialize vulkan helpers
  //  
  VkQueue computeQueue, transferQueue;
  {
    auto queueComputeFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue(device, queueComputeFID, 0, &computeQueue);
    vkGetDeviceQueue(device, queueComputeFID, 0, &transferQueue);
  }

  auto pCopyHelper = std::make_shared<vk_utils::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 8*1024*1024);
  auto pGPUImpl    = std::make_shared<ArrayProcess_GPU>();                // !!! USING GENERATED CODE !!!
  pGPUImpl->ReserveTestData(inArrayCPU.size());                           // !!! USING GENERATED CODE !!!
  pGPUImpl->InitVulkanObjects(device, physicalDevice, inArrayCPU.size()); // !!! USING GENERATED CODE !!!
  pGPUImpl->InitMemberBuffers();                                          // !!! USING GENERATED CODE !!! 

  // (3) Create buffer
  //
  const size_t bufferSize = inArrayCPU.size()*sizeof(uint32_t);
  VkBuffer numbersBuffer  = vk_utils::createBuffer(device, bufferSize,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer summBuffer     = vk_utils::createBuffer(device, bufferSize,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer prodBuffer     = vk_utils::createBuffer(device, bufferSize,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer reductBuffer   = vk_utils::createBuffer(device, 3*sizeof(uint32_t), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  
  VkDeviceMemory colorMem = vk_utils::allocateAndBindWithPadding(device, physicalDevice, {numbersBuffer, summBuffer, prodBuffer, reductBuffer});

  MyInOutVulkan userOut;
  userOut.summBuffer      = summBuffer;
  userOut.productBuffer   = prodBuffer;
  userOut.reductionBuffer = reductBuffer;
  userOut.someSize        = inArrayCPU.size();
  pGPUImpl->SetOutputVulkan(userOut);                                            // !!! USING GENERATED CODE !!!
  pGPUImpl->SetVulkanInOutFor_ProcessArrays(numbersBuffer, 0, numbersBuffer, 0); // !!! USING GENERATED CODE !!!
  pGPUImpl->UpdateAll(pCopyHelper);                                              // !!! USING GENERATED CODE !!!
  
  pCopyHelper->UpdateBuffer(numbersBuffer, 0, inArrayCPU.data(), bufferSize);

  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    pGPUImpl->ProcessArraysCmd(commandBuffer, nullptr, nullptr, inArrayCPU.size()); // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for command buffer execution " << std::endl;
    
    std::vector<int32_t> summRes(inArrayCPU.size()), prodRes(inArrayCPU.size()), reductionRes(3);
    
    pCopyHelper->ReadBuffer(summBuffer, 0, summRes.data(), summRes.size()*sizeof(int32_t));
    pCopyHelper->ReadBuffer(prodBuffer, 0, prodRes.data(), prodRes.size()*sizeof(int32_t));
    pCopyHelper->ReadBuffer(reductBuffer, 0, reductionRes.data(), reductionRes.size()*sizeof(int32_t));

    bool failed = false;
    for(size_t i=0;i<summRes.size();i++)
    {
      if(summRes[i] != inArrayCPU[i] + inArrayCPU[i] || 
         prodRes[i] != 222)
      {
        failed = true;
        break;
      }
    }

    if(failed)
      std::cout << "test FAILED!" << std::endl;
    else
      std::cout << "test PASSED!" << std::endl;
    
    std::cout << "summGPU = " << reductionRes[0] << std::endl;
    std::cout << "minvGPU = " << reductionRes[1] << std::endl;
    std::cout << "maxvGPU = " << reductionRes[2] << std::endl;

    //ArrayProcess_UBO_Data uboData;
    //pCopyHelper->ReadBuffer(pGPUImpl->GiveMeUBO(), 0, &uboData, sizeof(ArrayProcess_UBO_Data));
    //resSumm = uboData.m_summ;
  }
  
  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl    = nullptr;    // !!! USING GENERATED CODE !!! 

  vkDestroyBuffer(device, numbersBuffer, nullptr);
  vkDestroyBuffer(device, summBuffer, nullptr);
  vkDestroyBuffer(device, prodBuffer, nullptr);
  vkDestroyBuffer(device, reductBuffer, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);

  return resSumm;
}