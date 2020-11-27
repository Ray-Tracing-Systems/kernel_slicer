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

#include "vulkan_basics.h"
void TestKernel(VulkanContext a_vkData);
void TestKernel2(VulkanContext vk_data);

#include "test_class_generated.h"

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
  enabledLayers.push_back("VK_LAYER_LUNARG_standard_validation");
  instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers, extensions);

  physicalDevice       = vk_utils::FindPhysicalDevice(instance, true, 0);
  auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  vk_utils::queueFamilyIndices fIDs = {};

  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");

  fIDs.compute = queueComputeFID;
  device       = vk_utils::CreateLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures, fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  commandPool  = vk_utils::CreateCommandPool(device, physicalDevice, VK_QUEUE_COMPUTE_BIT, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

  // (2) initialize vulkan helpers
  //  
  VkQueue computeQueue, transferQueue;
  {
    auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT);
    vkGetDeviceQueue(device, queueComputeFID, 0, &computeQueue);
    vkGetDeviceQueue(device, queueComputeFID, 0, &transferQueue);
  }

  VulkanContext ctx;
  ctx.instance       = instance;
  ctx.physicalDevice = physicalDevice;
  ctx.device         = device;
  ctx.commandPool    = commandPool;
  ctx.computeQueue   = computeQueue;
  ctx.transferQueue  = transferQueue;
  //TestKernel(ctx);
  TestKernel2(ctx);
  
  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 8*1024*1024);

  auto pGPUImpl = std::make_shared<TestClass_Generated>();     // !!! USING GENERATED CODE !!! 
  pGPUImpl->InitVulkanObjects(device, physicalDevice, WIN_WIDTH*WIN_HEIGHT, 16, 16, 1); // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSize  = WIN_WIDTH*WIN_HEIGHT*sizeof(uint32_t);
  VkBuffer colorBuffer     = vkfw::CreateBuffer(device, bufferSize,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkDeviceMemory colorMem  = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {colorBuffer});
  
  pGPUImpl->SetVulkanInputOutput(colorBuffer, 0);              // !!! USING GENERATED CODE !!! 
  pGPUImpl->UpdateAll(pCopyHelper);                            // !!! USING GENERATED CODE !!!

  // (4) fill buffer with yellow color
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool, 1)[0];
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);

    //vkCmdFillBuffer(commandBuffer, colorBuffer, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->MainFuncCmd(commandBuffer, WIN_WIDTH, WIN_HEIGHT, nullptr);  // !!! USING GENERATED CODE !!! 

    vkEndCommandBuffer(commandBuffer);  
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    float ms  = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for full command buffer execution " << std::endl;
  }
  
  // (5) get data back to CPU and save image to file
  //
  std::vector<uint32_t> pixelData(WIN_WIDTH*WIN_HEIGHT);
  pCopyHelper->ReadBuffer(colorBuffer, 0, pixelData.data(), pixelData.size()*sizeof(uint32_t));
  SaveBMP("zout_gpu.bmp", pixelData.data(), WIN_WIDTH, WIN_HEIGHT);

  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;

  vkDestroyBuffer(device, colorBuffer, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}