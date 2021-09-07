#include "Bitmap.h"

#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_descriptor_sets.h"
#include "vk_copy.h"
#include "vk_buffers.h"
#include "vk_images.h"

#include "vulkan_basics.h"
#include "test_class_generated.h"

void SaveTestImage(const float4* data, int w, int h);

void tone_mapping_gpu(int w, int h, const float* a_hdrData, const char* a_outName)
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

  auto pCopyHelper = std::make_shared<vk_utils::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 64*1024*1024);

  auto pGPUImpl = std::make_shared<ToneMapping_Generated>(); // !!! USING GENERATED CODE !!! 
  pGPUImpl->InitVulkanObjects(device, physicalDevice, w*h);  // !!! USING GENERATED CODE !!!
  pGPUImpl->SetSize(w, h);                                   // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  pGPUImpl->InitMemberBuffers();                             // !!! USING GENERATED CODE !!!
  pGPUImpl->UpdateAll(pCopyHelper);                          // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSizeLDR = w*h*sizeof(uint32_t);
  VkBuffer colorBufferLDR    = vk_utils::createBuffer(device, bufferSizeLDR,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  VkDeviceMemory colorMem    = vk_utils::allocateAndBindWithPadding(device, physicalDevice, {colorBufferLDR});
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////// create input texture
  vk_utils::VulkanImageMem imgMem = {};

  imgMem = vk_utils::createImg(device, w, h, VK_FORMAT_R32G32B32A32_SFLOAT,
                               VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT);
  imgMem.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;

  {
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = imgMem.memReq.size;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(imgMem.memReq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &imgMem.mem));
  }

  vk_utils::createImageViewAndBindMem(device, &imgMem);
  pCopyHelper->UpdateImage(imgMem.image, a_hdrData, w, h, sizeof(float) * 4);

  {
    auto imgCmdBuf = vk_utils::createCommandBuffer(device, commandPool);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(imgCmdBuf, &beginInfo);
    {
      VkImageSubresourceRange subresourceRange = {};
      subresourceRange.aspectMask = imgMem.aspectMask;
      subresourceRange.levelCount = 1;
      subresourceRange.layerCount = 1;
      vk_utils::setImageLayout(
          imgCmdBuf,
          imgMem.image,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
          VK_IMAGE_LAYOUT_GENERAL,
          subresourceRange);
    }
    vkEndCommandBuffer(imgCmdBuf);
    vk_utils::executeCommandBufferNow(imgCmdBuf, transferQueue, device);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////// \\\ end of create input texture

  pGPUImpl->SetVulkanInOutFor_Bloom(imgMem.image, imgMem.view, colorBufferLDR, 0);
  
  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(device, commandPool);
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    //vkCmdFillBuffer(commandBuffer, colorBufferLDR, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->BloomCmd(commandBuffer, w, h, Texture2D<float4>(), nullptr);         // !!! USING GENERATED CODE !!! 
   
    vkEndCommandBuffer(commandBuffer);  
    
    float minTime = 1.0e20f;
    for(int i=0;i<10;i++)
    {
      auto start = std::chrono::high_resolution_clock::now();
      vk_utils::executeCommandBufferNow(commandBuffer, computeQueue, device);
      auto stop = std::chrono::high_resolution_clock::now();
      auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
      minTime   = std::min(minTime, ms);
    }
    std::cout << minTime << " ms for command buffer execution " << std::endl;

    std::vector<unsigned int> pixels(w*h);
    pCopyHelper->ReadBuffer(colorBufferLDR, 0, pixels.data(), pixels.size()*sizeof(unsigned int));
    SaveBMP(a_outName, pixels.data(), w, h);

    std::cout << std::endl;
  }
  
  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!! 
//  inputTex = nullptr;

  vkDestroyBuffer(device, colorBufferLDR, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyImageView(device, imgMem.view, nullptr);
  vkDestroyImage(device, imgMem.image, nullptr);
  vkFreeMemory(device, imgMem.mem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}