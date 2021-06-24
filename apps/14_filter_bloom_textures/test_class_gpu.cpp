#include "Bitmap.h"

#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "vk_utils.h"
#include "vk_program.h"
#include "vk_copy.h"
#include "vk_buffer.h"
#include "vk_texture.h"

#include "vulkan_basics.h"
#include "test_class_generated.h"

//class ToneMapping_Debug : public ToneMapping_Generated
//{
//public:
//  
//  void SaveTestImageNow(const char* a_outName, std::shared_ptr<vkfw::ICopyEngine> a_pCopyEngine)
//  {
//    std::vector<float4> realColor(m_width*m_height);    
//    std::vector<unsigned int> pixels(m_width*m_height);
//
//    //a_pCopyEngine->ReadBuffer(colorBufferLDR, 0, pixels.data(), pixels.size()*sizeof(unsigned int));
//    a_pCopyEngine->ReadBuffer(m_vdata.m_brightPixelsBuffer, 0, realColor.data(), realColor.size()*sizeof(float4));
//
//    for(int i=0;i<pixels.size();i++)
//      pixels[i] = RealColorToUint32(clamp(realColor[i], 0.0f, 1.0f));
//    
//    SaveBMP(a_outName, pixels.data(), m_width, m_height);
//  }
//};

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
  instance = vk_utils::CreateInstance(enableValidationLayers, enabledLayers, extensions);
  volkLoadInstance(instance);

  physicalDevice       = vk_utils::FindPhysicalDevice(instance, true, 0);
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

  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(physicalDevice, device, transferQueue, queueComputeFID, 64*1024*1024);

  auto pGPUImpl = std::make_shared<ToneMapping_Generated>(); // !!! USING GENERATED CODE !!! 
  pGPUImpl->InitVulkanObjects(device, physicalDevice, w*h);  // !!! USING GENERATED CODE !!!
  pGPUImpl->SetSize(w, h);                                   // must initialize all vector members with correct capacity before call 'InitMemberBuffers()'
  pGPUImpl->InitMemberBuffers();                             // !!! USING GENERATED CODE !!!
  pGPUImpl->UpdateAll(pCopyHelper);                          // !!! USING GENERATED CODE !!!

  // (3) Create buffer
  //
  const size_t bufferSizeLDR = w*h*sizeof(uint32_t);
  VkBuffer colorBufferLDR    = vkfw::CreateBuffer(device, bufferSizeLDR,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);

  VkDeviceMemory colorMem    = vkfw::AllocateAndBindWithPadding(device, physicalDevice, {colorBufferLDR});
  
  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////// create input texture
  vkfw::ImageParameters parameters;
  parameters.format       = VK_FORMAT_R32G32B32A32_SFLOAT;
  parameters.width        = w;
  parameters.height       = h;
  parameters.mipLevels    = 1;
  parameters.filterable   = true;
  parameters.renderable   = false;
  parameters.transferable = true;
  parameters.loadstore    = true;
  auto inputTex  = std::make_shared<vkfw::SimpleTexture2D>();
  auto memReqTex = inputTex->CreateImage(device, parameters);

  // memory for all read-only textures
  VkDeviceMemory memInputTex = VK_NULL_HANDLE;
  {
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = memReqTex.size;
    allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memReqTex.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, physicalDevice);
    VK_CHECK_RESULT(vkAllocateMemory(device, &allocateInfo, NULL, &memInputTex));
  }
  
  inputTex->BindMemory(memInputTex, 0);
  inputTex->Update(a_hdrData, w, h, sizeof(float)*4, pCopyHelper.get()); // --> put inputTex in transfer_dst layout

  // transfer texture to shader_read layout
  {
    VkCommandBuffer cmdBuff = pCopyHelper->CmdBuffer();
    vkResetCommandBuffer(cmdBuff, 0);
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    if (vkBeginCommandBuffer(cmdBuff, &beginInfo) != VK_SUCCESS)
      throw std::runtime_error("[mip gen]: failed to begin command buffer!");

    inputTex->ChangeLayoutCmd(cmdBuff, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);  // --> put inputTex in shader_read layout
    
    vkEndCommandBuffer(cmdBuff);
    vk_utils::ExecuteCommandBufferNow(cmdBuff, transferQueue, device);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////// \\\ end of create input texture

  pGPUImpl->SetVulkanInOutFor_Bloom(inputTex->Image(), inputTex->View(), colorBufferLDR, 0);
  
  // now compute some thing useful
  //
  {
    VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(device, commandPool, 1)[0];
    
    VkCommandBufferBeginInfo beginCommandBufferInfo = {};
    beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
    vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);
    //vkCmdFillBuffer(commandBuffer, colorBufferLDR, 0, VK_WHOLE_SIZE, 0x0000FFFF); // fill with yellow color
    pGPUImpl->BloomCmd(commandBuffer, w, h, Texture2D<float4>(), nullptr);         // !!! USING GENERATED CODE !!! 
   
    vkEndCommandBuffer(commandBuffer);  
    
    auto start = std::chrono::high_resolution_clock::now();
    vk_utils::ExecuteCommandBufferNow(commandBuffer, computeQueue, device);
    auto stop = std::chrono::high_resolution_clock::now();
    auto ms   = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count()/1000.f;
    std::cout << ms << " ms for command buffer execution " << std::endl;

    std::vector<unsigned int> pixels(w*h);
    pCopyHelper->ReadBuffer(colorBufferLDR, 0, pixels.data(), pixels.size()*sizeof(unsigned int));
    SaveBMP(a_outName, pixels.data(), w, h);

    std::cout << std::endl;
  }
  
  // (6) destroy and free resources before exit
  //
  pCopyHelper = nullptr;
  pGPUImpl = nullptr;                                                       // !!! USING GENERATED CODE !!! 
  inputTex = nullptr;

  vkDestroyBuffer(device, colorBufferLDR, nullptr);
  vkFreeMemory(device, colorMem, nullptr);
  vkFreeMemory(device, memInputTex, nullptr);

  vkDestroyCommandPool(device, commandPool, nullptr);

  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
}