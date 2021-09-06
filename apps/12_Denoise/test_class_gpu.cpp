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

#include "vulkan_basics.h"
#include "test_class_generated.h"

void Denoise_gpu(const int w, const int h, const float* a_hdrData, int32_t* a_inTexColor, const int32_t* a_inNormal, const float* a_inDepth, 
                 const int a_windowRadius, const int a_blockRadius, const float a_noiseLevel, const char* a_outName)
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
  enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
  VK_CHECK_RESULT(volkInitialize());
  instance = vk_utils::createInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::findPhysicalDevice(instance, true, 0);
  auto queueComputeFID = vk_utils::getQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  // query for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  vk_utils::QueueFID_T fIDs = {};

  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 

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

  auto pGPUImpl = std::make_shared<Denoise_Generated>();     // !!! USING GENERATED CODE !!! 
  pGPUImpl->InitVulkanObjects(device, physicalDevice, w*h);  // !!! USING GENERATED CODE !!!

  pGPUImpl->Resize(w, h);                                    // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  pGPUImpl->InitMemberBuffers();                             // !!! USING GENERATED CODE !!!
  pGPUImpl->UpdateAll(pCopyHelper);                          // !!! USING GENERATED CODE !!!

  // (3) Create buffers
  //
  const size_t bufferSize1 = w*h*sizeof(uint32_t);
  const size_t bufferSize4 = w*h*sizeof(float)*4;

  VkBuffer buff_hdrData    = vk_utils::createBuffer(device, bufferSize4,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VkBuffer buff_inTexColor = vk_utils::createBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VkBuffer buff_inNormal   = vk_utils::createBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VkBuffer buff_inDepth    = vk_utils::createBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VkBuffer buff_outColor   = vk_utils::createBuffer(device, bufferSize1,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  VkDeviceMemory colorMem  = vk_utils::allocateAndBindWithPadding(device, physicalDevice, {buff_hdrData, buff_inTexColor, buff_inNormal, buff_inDepth, buff_outColor});

  pCopyHelper->UpdateBuffer(buff_hdrData   , 0, a_hdrData,    bufferSize4);
  pCopyHelper->UpdateBuffer(buff_inTexColor, 0, a_inTexColor, bufferSize1);
  pCopyHelper->UpdateBuffer(buff_inNormal  , 0, a_inNormal,   bufferSize1);
  pCopyHelper->UpdateBuffer(buff_inDepth   , 0, a_inDepth,    bufferSize1);

  pGPUImpl->SetVulkanInOutFor_NLM_denoise(buff_hdrData,    0,
                                          buff_outColor,   0,
                                          buff_inTexColor, 0,
                                          buff_inNormal,   0,
                                          buff_inDepth,    0);
  
  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    //vkCmdFillBuffer(commandBuffer, colorBufferLDR, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->NLM_denoiseCmd(commandBuffer, w, h, nullptr, nullptr, nullptr, nullptr, nullptr,  a_windowRadius, a_blockRadius, a_noiseLevel);
    vkEndCommandBuffer(commandBuffer);  
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for command buffer execution " << std::endl;

    std::vector<unsigned int> pixels(w*h);
    pCopyHelper->ReadBuffer(buff_outColor, 0, pixels.data(), pixels.size()*sizeof(unsigned int));
    SaveBMP(a_outName, pixels.data(), w, h);

    //pGPUImpl->SaveTestImageNow("z_test.bmp", pCopyHelper);

    std::cout << std::endl;
  }
  
  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!! 

  vkDestroyBuffer(device, buff_hdrData, nullptr);
  vkDestroyBuffer(device, buff_inTexColor, nullptr);
  vkDestroyBuffer(device, buff_inNormal, nullptr);
  vkDestroyBuffer(device, buff_inDepth, nullptr);
  vkDestroyBuffer(device, buff_outColor, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}