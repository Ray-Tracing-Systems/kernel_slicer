#include "Bitmap.h"

#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_program.h"
#include "vk_copy.h"
#include "vk_buffer.h"

#include "vulkan_basics.h"
#include "test_class_generated.h"

#include "include/SphHarm_ubo.h"

class SphHarm_GPU : public SphHarm_Generated
{
public:
  SphHarm_GPU(){}

  VkBufferUsageFlags GetAdditionalFlagsForUBO() const override { return VK_BUFFER_USAGE_TRANSFER_SRC_BIT; }
  VkBuffer           GiveMeUBO() { return m_classDataBuffer; }
  VkBuffer           GiveMeTempBuffer() { return m_vdata.tmpred012Buffer; }
};

std::array<LiteMath::float3, 9> process_image_gpu(std::vector<uint32_t>& a_inPixels, uint32_t a_width, uint32_t a_height)
{
  std::array<LiteMath::float3, 9> result;

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
  instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::FindPhysicalDevice(instance, true, 1);
  auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  // query for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;

  // query for VariablePointers
  //
  VkPhysicalDeviceVariablePointersFeatures varPointers = {};
  varPointers.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VARIABLE_POINTERS_FEATURES;
  varPointers.pNext = &features;
  varPointers.variablePointers              = VK_TRUE;
  varPointers.variablePointersStorageBuffer = VK_TRUE;
  
  VkPhysicalDeviceFeatures2 physDevFeatures2 = {};
  physDevFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  physDevFeatures2.pNext = &varPointers;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;
  vk_utils::queueFamilyIndices fIDs = {};

  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 

  fIDs.compute = queueComputeFID;
  device       = vk_utils::CreateLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures, 
                                               fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT, physDevFeatures2);
  volkLoadDevice(device);
                                              
  commandPool  = vk_utils::CreateCommandPool(device, physicalDevice, VK_QUEUE_COMPUTE_BIT, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // (2) initialize vulkan helpers
  //  
  VkQueue computeQueue, transferQueue;
  {
    auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue(device, queueComputeFID, 0, &computeQueue);
    vkGetDeviceQueue(device, queueComputeFID, 0, &transferQueue);
  }

  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 8*1024*1024);

  auto pGPUImpl = std::make_shared<SphHarm_GPU>();                        // !!! USING GENERATED CODE !!! 
  pGPUImpl->InitVulkanObjects(device, physicalDevice, a_inPixels.size()); // !!! USING GENERATED CODE !!!
  pGPUImpl->InitMemberBuffers();                                          // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSizeLDR = a_inPixels.size()*sizeof(uint32_t);
  VkBuffer colorBufferLDR    = vkfw::CreateBuffer(device, bufferSizeLDR,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  VkDeviceMemory colorMem    = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {colorBufferLDR});

  pGPUImpl->SetVulkanInOutFor_ProcessPixels(colorBufferLDR, 0); // <==
  pGPUImpl->UpdateAll(pCopyHelper);                             // !!! USING GENERATED CODE !!!
  pCopyHelper->UpdateBuffer(colorBufferLDR, 0, a_inPixels.data(), bufferSizeLDR);

  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool, 1)[0];
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    pGPUImpl->ProcessPixelsCmd(commandBuffer, nullptr, a_width, a_height); // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for command buffer execution " << std::endl;

    SphHarm_UBO_Data uboData;
    pCopyHelper->ReadBuffer(pGPUImpl->GiveMeUBO(), 0, &uboData, sizeof(SphHarm_UBO_Data));
    memcpy(result.data(), uboData.coefs, result.size()*sizeof(float3));

    //std::vector<float3> tempSumm(3214); // ((a_width*a_height)/256);
    //pCopyHelper->ReadBuffer(pGPUImpl->GiveMeTempBuffer(), 0, tempSumm.data(), tempSumm.size()*sizeof(float3));
    //
    //std::ofstream fout("colors.txt");
    //uint32_t currSize   = (a_width*a_height)/256;
    //uint32_t currOffset = 0;
    //while(currSize > 1)
    //{
    //  for(uint32_t i=0;i<currSize;i++)
    //    fout << i << ":\t" << tempSumm[currOffset+i].x << " " << tempSumm[currOffset+i].y << " " << tempSumm[currOffset+i].z << std::endl;
    //  fout << "================================" << std::endl;
    //  currOffset += currSize;
    //  currSize = (currSize + 256 - 1) / 256;
    //}
  }
  
  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!! 

  vkDestroyBuffer(device, colorBufferLDR, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);

  return result;
}