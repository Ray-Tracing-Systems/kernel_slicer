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


#include "vk_compute_pipeline.h"

#include "vulkan_basics.h"
#include "test_class.h"

void TestKernel(VulkanContext vk_data)
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  VkPhysicalDeviceProperties devProps;
  vkGetPhysicalDeviceProperties(vk_data.physicalDevice, &devProps);

  auto queueComputeFID  = vk_utils::GetQueueFamilyIndex(vk_data.physicalDevice, VK_QUEUE_COMPUTE_BIT);
  auto queueTransferFID = vk_utils::GetQueueFamilyIndex(vk_data.physicalDevice, VK_QUEUE_TRANSFER_BIT);

  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(vk_data.physicalDevice, vk_data.device, vk_data.transferQueue, queueTransferFID, 8*1024*1024);
  auto pMaker      = std::make_shared<vkfw::ComputePipelineMaker>();

  VkDescriptorType dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  uint32_t dtypesize     = 16;
  auto pBindings   = std::make_shared<vkfw::ProgramBindings>(vk_data.device, &dtype, &dtypesize, 1);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VkBuffer rayPosBuffer   = vkfw::CreateBuffer(vk_data.device, WIN_WIDTH*WIN_HEIGHT*sizeof(float)*4,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer rayDirBuffer   = vkfw::CreateBuffer(vk_data.device, WIN_WIDTH*WIN_HEIGHT*sizeof(float)*4,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer rayFlagsBuffer = vkfw::CreateBuffer(vk_data.device, WIN_WIDTH*WIN_HEIGHT*sizeof(uint32_t),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer localsData     = vkfw::CreateBuffer(vk_data.device, sizeof(float)*16,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VkDeviceMemory mem = vkfw::AllocateAndBindWithPadding(vk_data.device, vk_data.physicalDevice, {rayPosBuffer, rayDirBuffer, rayFlagsBuffer, localsData});

  const float4x4 proj               = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
  const float4x4 mWorldViewProjInv  = inverse4x4(proj);
  pCopyHelper->UpdateBuffer(localsData, 0, &mWorldViewProjInv, sizeof(float)*16);
  
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VkSpecializationMapEntry specializationEntries[3] = {};
  {
    specializationEntries[0].constantID = 0;
    specializationEntries[0].offset     = 0;
    specializationEntries[0].size       = sizeof(uint32_t);
  
    specializationEntries[1].constantID = 1;
    specializationEntries[1].offset     = sizeof(uint32_t);
    specializationEntries[1].size       = sizeof(uint32_t);
  
    specializationEntries[2].constantID = 2;
    specializationEntries[2].offset     = 2 * sizeof(uint32_t);
    specializationEntries[2].size       = sizeof(uint32_t);
  }

  uint32_t specializationData[3] = {16, 16, 1};
  VkSpecializationInfo specsForWGSize = {};
  {
    specsForWGSize.mapEntryCount = 3;
    specsForWGSize.pMapEntries   = specializationEntries;
    specsForWGSize.dataSize      = 3 * sizeof(uint32_t);
    specsForWGSize.pData         = specializationData;
  }

  pMaker->CreateShader(vk_data.device, "z_generated.cl.spv", &specsForWGSize, "kernel_InitEyeRay");
  //pMaker->CreateShader(vk_data.device, "z_test.comp.spv", &specsForWGSIze, "main");
  
  VkDescriptorSet ds        = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE; 
  pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  pBindings->BindBuffer(0, rayFlagsBuffer);
  pBindings->BindBuffer(1, rayPosBuffer);
  pBindings->BindBuffer(2, rayDirBuffer);
  pBindings->BindBuffer(3, localsData);
  pBindings->BindEnd(&ds, &dsl);
  
  auto pipeL = pMaker->MakeLayout(vk_data.device, dsl, sizeof(uint32_t)*2);
  auto pipeO = pMaker->MakePipeline(vk_data.device);
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(vk_data.device, vk_data.commandPool, 1)[0];
  
  VkCommandBufferBeginInfo beginCommandBufferInfo = {};
  beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);

  vkCmdBindPipeline      (commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeO);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeL, 0, 1, &ds, 0, nullptr);
 
  uint32_t pcData[2] = {WIN_WIDTH, WIN_HEIGHT};
  vkCmdPushConstants(commandBuffer, pipeL, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)*2, pcData);
  vkCmdDispatch(commandBuffer, WIN_WIDTH/16, WIN_HEIGHT/16, 1);

  vkEndCommandBuffer(commandBuffer);

  vk_utils::ExecuteCommandBufferNow(commandBuffer, vk_data.computeQueue, vk_data.device);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<float4>  rayPosCPU(WIN_WIDTH*WIN_HEIGHT);
  std::vector<float4>  rayDirCPU(WIN_WIDTH*WIN_HEIGHT);
  std::vector<uint32_t> rayFlagsCPU(WIN_WIDTH*WIN_HEIGHT);

  pCopyHelper->ReadBuffer(rayPosBuffer, 0, rayPosCPU.data(), rayPosCPU.size()*sizeof(float4));
  pCopyHelper->ReadBuffer(rayDirBuffer, 0, rayDirCPU.data(), rayDirCPU.size()*sizeof(float4));
  pCopyHelper->ReadBuffer(rayFlagsBuffer, 0, rayFlagsCPU.data(), rayFlagsCPU.size()*sizeof(uint32_t));
  
  std::vector<float4>  rayDirCPU2(WIN_WIDTH*WIN_HEIGHT);
  for(int y=0;y<WIN_HEIGHT;y++)
  {
    for(int x=0;x<WIN_WIDTH;x++)
    {
      float3 rayDir = EyeRayDir((float)x, (float)y, (float)WIN_WIDTH, (float)WIN_HEIGHT, mWorldViewProjInv); 
      rayDirCPU2[pitchOffset(x,y)] = to_float4(rayDir, MAXFLOAT);
    }
  }

  bool failed = false;
  for(size_t i=0;i<rayDirCPU.size();i++)
  {
    const float3 v1 = to_float3(rayDirCPU[i]);
    const float3 v2 = to_float3(rayDirCPU2[i]);

    float len = length(v2-v1);
    if(len > 1e-6f)
    {
      failed = true;
      break;
    }
  }

  if(failed)
    std::cout << "FAILED!" << std::endl;
  else
    std::cout << "PASSED!" << std::endl;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  vkDestroyBuffer(vk_data.device, localsData, nullptr);
  vkDestroyBuffer(vk_data.device, rayPosBuffer, nullptr);
  vkDestroyBuffer(vk_data.device, rayDirBuffer, nullptr);
  vkDestroyBuffer(vk_data.device, rayFlagsBuffer, nullptr);
  vkFreeMemory(vk_data.device, mem, nullptr);
}

void TestKernel2(VulkanContext vk_data)
{
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  VkPhysicalDeviceProperties devProps;
  vkGetPhysicalDeviceProperties(vk_data.physicalDevice, &devProps);

  auto queueComputeFID  = vk_utils::GetQueueFamilyIndex(vk_data.physicalDevice, VK_QUEUE_COMPUTE_BIT);
  auto queueTransferFID = vk_utils::GetQueueFamilyIndex(vk_data.physicalDevice, VK_QUEUE_TRANSFER_BIT);

  auto pCopyHelper = std::make_shared<vkfw::SimpleCopyHelper>(vk_data.physicalDevice, vk_data.device, vk_data.transferQueue, queueTransferFID, 8*1024*1024);
  auto pMaker      = std::make_shared<vkfw::ComputePipelineMaker>();

  VkDescriptorType dtype = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  uint32_t dtypesize     = 16;
  auto pBindings   = std::make_shared<vkfw::ProgramBindings>(vk_data.device, &dtype, &dtypesize, 1);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VkBuffer colorBuffer = vkfw::CreateBuffer(vk_data.device, WIN_WIDTH*WIN_HEIGHT*sizeof(uint32_t),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer hitsBuffer  = vkfw::CreateBuffer(vk_data.device, WIN_WIDTH*WIN_HEIGHT*sizeof(Lite_Hit),  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VkBuffer localsData  = vkfw::CreateBuffer(vk_data.device, sizeof(float)*16,  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);

  VkDeviceMemory mem = vkfw::AllocateAndBindWithPadding(vk_data.device, vk_data.physicalDevice, {colorBuffer, hitsBuffer, localsData});

  const float4x4 proj               = perspectiveMatrix(90.0f, 1.0f, 0.1f, 100.0f);
  const float4x4 mWorldViewProjInv  = inverse4x4(proj);
  pCopyHelper->UpdateBuffer(localsData, 0, &mWorldViewProjInv, sizeof(float)*16);
  
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VkSpecializationMapEntry specializationEntries[3] = {};
  {
    specializationEntries[0].constantID = 0;
    specializationEntries[0].offset     = 0;
    specializationEntries[0].size       = sizeof(uint32_t);
  
    specializationEntries[1].constantID = 1;
    specializationEntries[1].offset     = sizeof(uint32_t);
    specializationEntries[1].size       = sizeof(uint32_t);
  
    specializationEntries[2].constantID = 2;
    specializationEntries[2].offset     = 2 * sizeof(uint32_t);
    specializationEntries[2].size       = sizeof(uint32_t);
  }

  uint32_t specializationData[3] = {16, 16, 1};
  VkSpecializationInfo specsForWGSize = {};
  {
    specsForWGSize.mapEntryCount = 3;
    specsForWGSize.pMapEntries   = specializationEntries;
    specsForWGSize.dataSize      = 3 * sizeof(uint32_t);
    specsForWGSize.pData         = specializationData;
  }

  pMaker->CreateShader(vk_data.device, "z_generated.cl.spv", &specsForWGSize, "kernel_TestColor");

  VkDescriptorSet ds        = VK_NULL_HANDLE;
  VkDescriptorSetLayout dsl = VK_NULL_HANDLE; 
  pBindings->BindBegin(VK_SHADER_STAGE_COMPUTE_BIT);
  pBindings->BindBuffer(0, hitsBuffer);
  pBindings->BindBuffer(1, colorBuffer);
  pBindings->BindBuffer(2, localsData);
  pBindings->BindEnd(&ds, &dsl);
  
  auto pipeL = pMaker->MakeLayout(vk_data.device, dsl, sizeof(uint32_t)*2);
  auto pipeO = pMaker->MakePipeline(vk_data.device);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  VkCommandBuffer commandBuffer = vk_utils::CreateCommandBuffers(vk_data.device, vk_data.commandPool, 1)[0];
  
  VkCommandBufferBeginInfo beginCommandBufferInfo = {};
  beginCommandBufferInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  beginCommandBufferInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;
  vkBeginCommandBuffer(commandBuffer, &beginCommandBufferInfo);

  vkCmdBindPipeline      (commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeO);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeL, 0, 1, &ds, 0, nullptr);
 
  uint32_t pcData[2] = {WIN_WIDTH, WIN_HEIGHT};
  vkCmdPushConstants(commandBuffer, pipeL, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)*2, pcData);
  vkCmdDispatch(commandBuffer, WIN_WIDTH/16, WIN_HEIGHT/16, 1);

  vkEndCommandBuffer(commandBuffer);

  vk_utils::ExecuteCommandBufferNow(commandBuffer, vk_data.computeQueue, vk_data.device);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  std::vector<uint32_t>  colorCPU(WIN_WIDTH*WIN_HEIGHT);
  pCopyHelper->ReadBuffer(colorBuffer, 0, colorCPU.data(), colorCPU.size()*sizeof(uint32_t));
  SaveBMP("zout_gpu2.bmp", colorCPU.data(), WIN_WIDTH, WIN_HEIGHT);

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  vkDestroyBuffer(vk_data.device, localsData, nullptr);
  vkDestroyBuffer(vk_data.device, colorBuffer, nullptr);
  vkDestroyBuffer(vk_data.device, hitsBuffer, nullptr);
  vkFreeMemory(vk_data.device, mem, nullptr);
}

