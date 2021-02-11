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

#include "include/RedPixels_ubo.h"

class RedPixels_GPU : public RedPixels_Generated
{
public:
  RedPixels_GPU(){}

  VkBufferUsageFlags GetAdditionalFlagsForUBO() override { return VK_BUFFER_USAGE_TRANSFER_SRC_BIT; }
  VkBuffer GiveMeUBO() { return m_classDataBuffer; }
  VkBuffer GiveMeTempBuffer() { return m_vdata.tmpred04Buffer; }
};

void process_image_gpu(const std::vector<uint32_t>& a_inPixels, std::vector<RedPixels::PixelInfo>& a_outPixels)
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

  physicalDevice       = vk_utils::FindPhysicalDevice(instance, true, 1);
  auto queueComputeFID = vk_utils::GetQueueFamilyIndex(physicalDevice, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT);
  
  // query for shaderInt8
  //
  VkPhysicalDeviceShaderFloat16Int8Features features = {};
  features.sType      = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
  features.shaderInt8 = VK_TRUE;
  
  VkPhysicalDeviceFeatures2 physDevFeatures2 = {};
  physDevFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  physDevFeatures2.pNext = &features;

  std::vector<const char*> validationLayers, deviceExtensions;
  VkPhysicalDeviceFeatures enabledDeviceFeatures = {};
  enabledDeviceFeatures.shaderInt64 = VK_TRUE;
  vk_utils::queueFamilyIndices fIDs = {};

  deviceExtensions.push_back("VK_KHR_shader_non_semantic_info");
  deviceExtensions.push_back("VK_KHR_shader_float16_int8"); 

  fIDs.compute = queueComputeFID;
  device       = vk_utils::CreateLogicalDevice(physicalDevice, validationLayers, deviceExtensions, enabledDeviceFeatures, 
                                               fIDs, VK_QUEUE_TRANSFER_BIT | VK_QUEUE_COMPUTE_BIT, physDevFeatures2);
                                              
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

  auto pGPUImpl = std::make_shared<RedPixels_GPU>();                                 // !!! USING GENERATED CODE !!! 
  pGPUImpl->InitVulkanObjects(device, physicalDevice, a_inPixels.size(), 256, 1, 1); // !!! USING GENERATED CODE !!!
  
  pGPUImpl->SetMaxDataSize(a_inPixels.size());                         // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  pGPUImpl->InitMemberBuffers();                                       // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSizeLDR = a_inPixels.size()*sizeof(uint32_t);
  VkBuffer colorBufferLDR    = vkfw::CreateBuffer(device, bufferSizeLDR,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer colorBufferOUT    = vkfw::CreateBuffer(device, bufferSizeLDR,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  VkDeviceMemory colorMem    = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {colorBufferLDR, colorBufferOUT});

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
    //vkCmdFillBuffer(commandBuffer, colorBufferLDR, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->ProcessPixelsCmd(commandBuffer, nullptr, a_inPixels.size()); // !!! USING GENERATED CODE !!! 
    vkEndCommandBuffer(commandBuffer);  
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for command buffer execution " << std::endl;

    RedPixels_UBO_Data uboData;
    pCopyHelper->ReadBuffer(pGPUImpl->GiveMeUBO(), 0, &uboData, sizeof(RedPixels_UBO_Data));

    std::cout << "[gpu]: m_redPixelsNum     = " << uboData.m_redPixelsNum << std::endl;
    std::cout << "[gpu]: m_otherPixelsNum   = " << uboData.m_otherPixelsNum << std::endl;
    std::cout << "[gpu]: m_testPixelsAmount = " << uboData.m_testPixelsAmount << std::endl;
    std::cout << "[gpu]: m_foundPixels_size = " << uboData.m_foundPixels_size << std::endl;
    std::cout << "[gpu]: m_testMin(float)   = " << uboData.m_testMin << std::endl;
    std::cout << "[gpu]: m_testMax(float)   = " << uboData.m_testMax << std::endl;

    //std::vector<float> fredBufferData(1024+4+1);
    //pCopyHelper->ReadBuffer(pGPUImpl->GiveMeTempBuffer(), 0, fredBufferData.data(), sizeof(float)*fredBufferData.size());
    //
    //std::ofstream fout("z_out.txt");
    //for(size_t i=0;i<fredBufferData.size();i++)
    //fout << i << ":\t" << fredBufferData[i] << std::endl;
    //fout.close();
    
    //std::vector<unsigned int> pixels(w*h);
    //pCopyHelper->ReadBuffer(colorBufferOUT, 0, pixels.data(), pixels.size()*sizeof(unsigned int));
    //SaveBMP(a_outName, pixels.data(), w, h);

    std::cout << std::endl;
  }
  
  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!! 

  vkDestroyBuffer(device, colorBufferLDR, nullptr);
  vkDestroyBuffer(device, colorBufferOUT, nullptr);
  vkFreeMemory(device, colorMem, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}