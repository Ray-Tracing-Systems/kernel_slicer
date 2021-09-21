#include "vk_copy.h"
#include "vk_utils.h"
#include "vk_buffers.h"

#include <cstring>
#include <cassert>
#include <iostream>

#include <cmath>
#include <cassert>

#include <algorithm>
#ifdef WIN32
#undef min
#undef max
#endif 


vk_utils::SimpleCopyHelper::SimpleCopyHelper()
{
  queue   = VK_NULL_HANDLE;
  cmdPool = VK_NULL_HANDLE;
  physDev = VK_NULL_HANDLE;
  dev     = VK_NULL_HANDLE;
  cmdBuff = VK_NULL_HANDLE;
  stagingSize = 0u;
  
  stagingBuff       = VK_NULL_HANDLE;
  stagingBuffMemory = VK_NULL_HANDLE;
}

vk_utils::SimpleCopyHelper::SimpleCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device,
                                         VkQueue a_transferQueue, uint32_t a_transferQueueIDX, size_t a_stagingBuffSize)
{
  physDev = a_physicalDevice;
  dev     = a_device;
  queue   = a_transferQueue;

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = a_transferQueueIDX;
  VK_CHECK_RESULT(vkCreateCommandPool(a_device, &poolInfo, nullptr, &cmdPool));

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool        = cmdPool;
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &allocInfo, &cmdBuff));

  vk_utils::createBufferStaging(a_device, a_physicalDevice, a_stagingBuffSize, stagingBuff, stagingBuffMemory);

  stagingSize = a_stagingBuffSize;
}

vk_utils::SimpleCopyHelper::~SimpleCopyHelper()
{
  if(stagingBuff != VK_NULL_HANDLE)
    vkDestroyBuffer(dev, stagingBuff, NULL);

  vkFreeMemory   (dev, stagingBuffMemory, NULL);

  vkFreeCommandBuffers(dev, cmdPool, 1, &cmdBuff);
  vkDestroyCommandPool(dev, cmdPool, nullptr);
}


void vk_utils::SimpleCopyHelper::UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size)
{
  assert(a_dstOffset % 4 == 0);
  assert(a_size      % 4 == 0);

  if (a_size <= SMALL_BUFF)
  {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkResetCommandBuffer(cmdBuff, 0);
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
    vkCmdUpdateBuffer   (cmdBuff, a_dst, a_dstOffset, a_size, a_src);
    vkEndCommandBuffer  (cmdBuff);
    vk_utils::executeCommandBufferNow(cmdBuff, queue, dev);
    return;
  }

  for(size_t currPos = 0; currPos < a_size; currPos += stagingSize)
  {
    size_t currCopySize = std::min(a_size - currPos, stagingSize);
    void* mappedMemory = nullptr;
    vkMapMemory(dev, stagingBuffMemory, 0, currCopySize, 0, &mappedMemory);
    memcpy(mappedMemory, (char*)(a_src) + currPos, currCopySize);
    vkUnmapMemory(dev, stagingBuffMemory);

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkResetCommandBuffer(cmdBuff, 0);
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
     
    VkBufferCopy region0 = {};
    region0.srcOffset    = 0;
    region0.dstOffset    = a_dstOffset + currPos;
    region0.size         = currCopySize;

    vkCmdCopyBuffer(cmdBuff, stagingBuff, a_dst, 1, &region0);

    vkEndCommandBuffer(cmdBuff);
    vk_utils::executeCommandBufferNow(cmdBuff, queue, dev);
  }
}

void vk_utils::SimpleCopyHelper::ReadBuffer(VkBuffer a_src, size_t a_srcOffset, void* a_dst, size_t a_size)
{
  assert(a_srcOffset % 4 == 0);
  assert(a_size      % 4 == 0);

  for(size_t currPos = 0; currPos < a_size; currPos += stagingSize)
  {
    size_t currCopySize = std::min(a_size - currPos, stagingSize);
    
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  
    vkResetCommandBuffer(cmdBuff, 0);
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
    VkBufferCopy region0 = {};
    region0.srcOffset    = a_srcOffset + currPos;
    region0.dstOffset    = 0;
    region0.size         = currCopySize;
    vkCmdCopyBuffer(cmdBuff, a_src, stagingBuff, 1, &region0);  
    vkEndCommandBuffer(cmdBuff);

    vk_utils::executeCommandBufferNow(cmdBuff, queue, dev);

    void* mappedMemory = nullptr;
    vkMapMemory(dev, stagingBuffMemory, 0, currCopySize, 0, &mappedMemory);
    memcpy((char*)(a_dst) + currPos, mappedMemory, currCopySize);
    vkUnmapMemory(dev, stagingBuffMemory); 
  }
}

void vk_utils::SimpleCopyHelper::UpdateImage(VkImage a_image, const void* a_src, int a_width, int a_height, int a_bpp)
{
  size_t a_size = a_width * a_height * a_bpp;

  void* mappedMemory = nullptr;
  vkMapMemory(dev, stagingBuffMemory, 0, a_size, 0, &mappedMemory);
  memcpy(mappedMemory, a_src, a_size);
  vkUnmapMemory(dev, stagingBuffMemory);


  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  vkResetCommandBuffer(cmdBuff, 0);
  vkBeginCommandBuffer(cmdBuff, &beginInfo);

  VkImageMemoryBarrier imgBar = {};
  {
    imgBar.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    imgBar.pNext = nullptr;
    imgBar.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    imgBar.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

    imgBar.srcAccessMask = 0;
    imgBar.dstAccessMask = 0;
    imgBar.oldLayout     = VK_IMAGE_LAYOUT_UNDEFINED;
    imgBar.newLayout     = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    imgBar.image         = a_image;

    imgBar.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
    imgBar.subresourceRange.baseMipLevel   = 0;
    imgBar.subresourceRange.levelCount     = 1;
    imgBar.subresourceRange.baseArrayLayer = 0;
    imgBar.subresourceRange.layerCount     = 1;
  };

  vkCmdPipelineBarrier(cmdBuff,
                       VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0,
                       0, nullptr,
                       0, nullptr,
                       1, &imgBar);


  VkImageSubresourceLayers shittylayers = {};
  shittylayers.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  shittylayers.mipLevel       = 0;
  shittylayers.baseArrayLayer = 0;
  shittylayers.layerCount     = 1;

  VkBufferImageCopy wholeRegion = {};
  wholeRegion.bufferOffset      = 0;
  wholeRegion.bufferRowLength   = uint32_t(a_width);
  wholeRegion.bufferImageHeight = uint32_t(a_height);
  wholeRegion.imageExtent       = VkExtent3D{ uint32_t(a_width), uint32_t(a_height), 1 };
  wholeRegion.imageOffset       = VkOffset3D{ 0,0,0 };
  wholeRegion.imageSubresource  = shittylayers;

  vkCmdCopyBufferToImage(cmdBuff, stagingBuff, a_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &wholeRegion);

  vkEndCommandBuffer(cmdBuff);

  vk_utils::executeCommandBufferNow(cmdBuff, queue, dev);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vk_utils::PingPongCopyHelper::PingPongCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device, VkQueue a_transferQueue,
  uint32_t a_transferQueueIDX, size_t a_stagingBuffSize) : SimpleCopyHelper()
{
  physDev = a_physicalDevice;
  dev     = a_device;
  queue   = a_transferQueue;

  VkCommandPoolCreateInfo poolInfo = {};
  poolInfo.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  poolInfo.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  poolInfo.queueFamilyIndex = a_transferQueueIDX;
  VK_CHECK_RESULT(vkCreateCommandPool(a_device, &poolInfo, nullptr, &cmdPool));

  VkCommandBufferAllocateInfo allocInfo = {};
  allocInfo.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocInfo.commandPool        = cmdPool;
  allocInfo.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  allocInfo.commandBufferCount = 1;
  VK_CHECK_RESULT(vkAllocateCommandBuffers(a_device, &allocInfo, &cmdBuff));

  stagingSize     = a_stagingBuffSize;
  stagingSizeHalf = a_stagingBuffSize/2; 

  {
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size        = stagingSizeHalf;
    bufferCreateInfo.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT; 
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, &staging[0]));
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, &staging[1]));

    bufferCreateInfo.size = stagingSize;
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, NULL, &stagingBuff));
    
    VkMemoryRequirements memoryRequirements, memoryRequirement2, memoryRequirement3;
    vkGetBufferMemoryRequirements(a_device, staging[0],  &memoryRequirements);
    vkGetBufferMemoryRequirements(a_device, staging[1],  &memoryRequirement2);
    vkGetBufferMemoryRequirements(a_device, stagingBuff, &memoryRequirement3);
      
    assert(memoryRequirements.size == memoryRequirement2.size);
    assert(memoryRequirements.size + memoryRequirement2.size == memoryRequirement3.size);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize  = memoryRequirements.size * 2;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, physDev);
    VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, &stagingBuffMemory));

    VK_CHECK_RESULT(vkBindBufferMemory(a_device, staging[0], stagingBuffMemory, 0));
    VK_CHECK_RESULT(vkBindBufferMemory(a_device, staging[1], stagingBuffMemory, stagingSizeHalf));
    VK_CHECK_RESULT(vkBindBufferMemory(a_device, stagingBuff, stagingBuffMemory, 0));
  }

  VkFenceCreateInfo fenceCreateInfo = {};
  fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fenceCreateInfo.flags = 0;
  VK_CHECK_RESULT(vkCreateFence(a_device, &fenceCreateInfo, NULL, &fence));
}

vk_utils::PingPongCopyHelper::~PingPongCopyHelper()
{
  vkDestroyFence (dev, fence, NULL);
  vkDestroyBuffer(dev, staging[0], NULL);
  vkDestroyBuffer(dev, staging[1], NULL);
}

void vk_utils::PingPongCopyHelper::SubmitCopy(VkBuffer a_dst, size_t a_dstOffset, size_t a_size, int a_currStagingId)
{
  VkBuffer readyBuff = staging[a_currStagingId];
  vkResetFences(dev, 1, &fence);

  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  vkResetCommandBuffer(cmdBuff, 0);
  vkBeginCommandBuffer(cmdBuff, &beginInfo);

  VkBufferCopy region0 = {};
  region0.srcOffset    = 0;
  region0.dstOffset    = a_dstOffset;
  region0.size         = a_size;
  vkCmdCopyBuffer(cmdBuff, readyBuff, a_dst, 1, &region0);
  vkEndCommandBuffer(cmdBuff);
  
  VkSubmitInfo submitInfo       = {};
  submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;        
  submitInfo.pCommandBuffers    = &cmdBuff; 
  VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
}

void vk_utils::PingPongCopyHelper::UpdateBuffer(VkBuffer a_dst, size_t a_dstOffset, const void* a_src, size_t a_size)
{
  assert(a_dstOffset % 4 == 0);
  assert(a_size      % 4 == 0);

  if (a_size <= SMALL_BUFF)
  {
    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    vkResetCommandBuffer(cmdBuff, 0);
    vkBeginCommandBuffer(cmdBuff, &beginInfo);
    vkCmdUpdateBuffer   (cmdBuff, a_dst, a_dstOffset, a_size, a_src);
    vkEndCommandBuffer  (cmdBuff);
    
    vkResetFences(dev, 1, &fence);
    VkSubmitInfo submitInfo       = {};
    submitInfo.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;        
    submitInfo.pCommandBuffers    = &cmdBuff; 
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
    VK_CHECK_RESULT(vkWaitForFences(dev, 1, &fence, VK_TRUE, vk_utils::DEFAULT_TIMEOUT));

    return;
  }

  uint32_t currStaging = 0;
  size_t currPos  = 0;
  size_t prevCopySize = 0;

  for(; currPos < a_size; currPos += stagingSizeHalf) // use ping-pong shceme
  {
    size_t currCopySize = std::min(a_size - currPos, stagingSizeHalf);

    // (0) begin (copy staging[prev] ==> result) in parallel with further vkMapMemory/memcpy/vkUnmapMemory
    //
    if(currPos != 0) 
      SubmitCopy(a_dst, a_dstOffset + currPos - stagingSizeHalf, prevCopySize, 1 - currStaging);
    
    // (1) (copy src ==> staging[curr])
    //
    void* mappedMemory = nullptr;
    vkMapMemory(dev, stagingBuffMemory, currStaging * stagingSizeHalf, currCopySize, 0, &mappedMemory);
    memcpy(mappedMemory, ((char*)(a_src)) + currPos, currCopySize);
    vkUnmapMemory(dev, stagingBuffMemory);
    
    // (3) end (staging[prev] ==> result)
    //
    if(currPos != 0) 
      vkWaitForFences(dev, 1, &fence, VK_TRUE, vk_utils::DEFAULT_TIMEOUT);

    currStaging  = 1 - currStaging;
    prevCopySize = currCopySize;
  }

  // last iter copy: (staging[prev] ==> result)
  //
  SubmitCopy(a_dst, a_dstOffset + currPos - stagingSizeHalf, prevCopySize, 1-currStaging);
  VK_CHECK_RESULT(vkWaitForFences(dev, 1, &fence, VK_TRUE, vk_utils::DEFAULT_TIMEOUT));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

vk_utils::ComputeCopyHelper::ComputeCopyHelper(VkPhysicalDevice a_physicalDevice, VkDevice a_device,
                                           VkQueue a_transferQueue, uint32_t a_transferQueueIDX,
                                           size_t a_stagingBuffSize, const char *a_csCopyPath,
                                           size_t a_auxBuffSize) : Base(a_physicalDevice, a_device, a_transferQueue,
                                                                        a_transferQueueIDX, a_stagingBuffSize)
{
  if(a_auxBuffSize == size_t(-1))
    a_auxBuffSize = a_stagingBuffSize;
  
  // (1) create aux 'VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT' buffer
  //
  {
    auxBuffMemory = VK_NULL_HANDLE;
    auxBuff       = VK_NULL_HANDLE;
  
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size        = a_auxBuffSize;
    bufferCreateInfo.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, nullptr, &auxBuff));
  
    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(a_device, auxBuff, &memoryRequirements);
  
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = a_auxBuffSize;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physicalDevice);
    
    VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, NULL, &auxBuffMemory));

    VK_CHECK_RESULT(vkBindBufferMemory(a_device, auxBuff, auxBuffMemory, 0));
  }

  // (2) we can not write DescriptorSet until we know the input buffer, but we can make layouts and pool and allocate DS.
  //
  {
    copyDescriptorSet       = VK_NULL_HANDLE;
    copyDescriptorSetLayout = VK_NULL_HANDLE;
    
    VkDescriptorSetLayoutBinding descriptorSetLayoutBinding[2] = {};
    descriptorSetLayoutBinding[0].binding         = 0; //
    descriptorSetLayoutBinding[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[0].descriptorCount = 1;
    descriptorSetLayoutBinding[0].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    descriptorSetLayoutBinding[1].binding         = 1; 
    descriptorSetLayoutBinding[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorSetLayoutBinding[1].descriptorCount = 1;
    descriptorSetLayoutBinding[1].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
  
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 2; 
    descriptorSetLayoutCreateInfo.pBindings    = descriptorSetLayoutBinding;
  
    // Create the descriptor set layout.
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(a_device, &descriptorSetLayoutCreateInfo, NULL, &copyDescriptorSetLayout));

    // now ds pool and ds itself
    //
    VkDescriptorPoolSize descriptorPoolSize;
    descriptorPoolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 2;

    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
    descriptorPoolCreateInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets       = 1; // we need to allocate at least 1 descriptor set
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes    = &descriptorPoolSize;

    // create descriptor pool.
    VK_CHECK_RESULT(vkCreateDescriptorPool(a_device, &descriptorPoolCreateInfo, NULL, &m_dsPool));

    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {};
    descriptorSetAllocateInfo.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocateInfo.descriptorPool     = m_dsPool; // pool to allocate from.
    descriptorSetAllocateInfo.descriptorSetCount = 1;        // allocate a descriptor set for two buffer
    descriptorSetAllocateInfo.pSetLayouts        = &copyDescriptorSetLayout;

    // allocate descriptor set.
    VK_CHECK_RESULT(vkAllocateDescriptorSets(a_device, &descriptorSetAllocateInfo, &copyDescriptorSet));
  }

  // (3) load shader and create compute pipeline
  //
  { 
    copyPipeline       = VK_NULL_HANDLE;
    copyPipelineLayout = VK_NULL_HANDLE;
  
    std::vector<uint32_t> code = vk_utils::readSPVFile(a_csCopyPath);
    assert(!code.empty());
  
    VkShaderModuleCreateInfo createInfo = {};
    createInfo.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode    = code.data();
    createInfo.codeSize = code.size()*sizeof(uint32_t);
  
    VK_CHECK_RESULT(vkCreateShaderModule(a_device, &createInfo, NULL, &csCopy));
  
    
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {};
    shaderStageCreateInfo.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = csCopy;
    shaderStageCreateInfo.pName  = "main";
    
    VkPushConstantRange pcRange = {};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset     = 0;
    pcRange.size       = sizeof(int);  
  
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
    pipelineLayoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts    = &copyDescriptorSetLayout;
    pipelineLayoutCreateInfo.pPushConstantRanges    = &pcRange; 
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &pipelineLayoutCreateInfo, NULL, &copyPipelineLayout));
  
    VkComputePipelineCreateInfo pipelineCreateInfo = {};
    pipelineCreateInfo.sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage  = shaderStageCreateInfo;
    pipelineCreateInfo.layout = copyPipelineLayout;
  
    VK_CHECK_RESULT(vkCreateComputePipelines(a_device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &copyPipeline));    
  }


}

vk_utils::ComputeCopyHelper::~ComputeCopyHelper()
{
  vkDestroyDescriptorPool(dev, m_dsPool, nullptr);
  vkFreeMemory(dev, auxBuffMemory, nullptr);
  vkDestroyBuffer(dev, auxBuff, nullptr);
  vkDestroyShaderModule(dev, csCopy, nullptr);
}


void vk_utils::ComputeCopyHelper::ReadBuffer(VkBuffer a_src, size_t a_srcOffset, void* a_dst, size_t a_size)
{
  // (1) bind input and output buffers
  //
  VkDescriptorBufferInfo descriptorBufferInfo[2] = {};
  {
    descriptorBufferInfo[0].buffer = a_src;
    descriptorBufferInfo[0].offset = a_srcOffset;
    descriptorBufferInfo[0].range  = a_size;
  
    descriptorBufferInfo[1].buffer = auxBuff;
    descriptorBufferInfo[1].offset = 0;
    descriptorBufferInfo[1].range  = a_size;
  }

  VkWriteDescriptorSet writeDescriptorSet[2] = {};
  {
    writeDescriptorSet[0].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[0].dstSet          = copyDescriptorSet; 
    writeDescriptorSet[0].dstBinding      = 0;                 
    writeDescriptorSet[0].descriptorCount = 1;                 
    writeDescriptorSet[0].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage bufferStaging.
    writeDescriptorSet[0].pBufferInfo     = &descriptorBufferInfo[0];
  
    writeDescriptorSet[1].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writeDescriptorSet[1].dstSet          = copyDescriptorSet; 
    writeDescriptorSet[1].dstBinding      = 1;                 
    writeDescriptorSet[1].descriptorCount = 1;                 
    writeDescriptorSet[1].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; // storage bufferStaging.
    writeDescriptorSet[1].pBufferInfo     = &descriptorBufferInfo[1];
  }
  // perform the update of the descriptor set.
  vkUpdateDescriptorSets(dev, 2, writeDescriptorSet, 0, NULL);


  // first, copy data to staging buff
  //
  VkCommandBufferBeginInfo beginInfo = {};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

  vkResetCommandBuffer(cmdBuff, 0);
  vkBeginCommandBuffer(cmdBuff, &beginInfo);


  vkCmdBindPipeline      (cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, copyPipeline);
  vkCmdBindDescriptorSets(cmdBuff, VK_PIPELINE_BIND_POINT_COMPUTE, copyPipelineLayout, 0, 1, &copyDescriptorSet, 0, NULL);

  int numThreads = int(a_size / (sizeof(float)*4) );
  vkCmdPushConstants(cmdBuff, copyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(int), &numThreads);

  uint32_t numGroups = uint32_t(numThreads/256);
  if(numGroups < 1) 
    numGroups = 1;

  vkCmdDispatch(cmdBuff, numGroups, 1, 1);
   
  VkBufferMemoryBarrier bufBarr = {};
  bufBarr.sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  bufBarr.pNext               = nullptr;
  bufBarr.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufBarr.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  bufBarr.size                = VK_WHOLE_SIZE;
  bufBarr.offset              = 0;
  bufBarr.buffer              = auxBuff;
  bufBarr.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
  bufBarr.dstAccessMask       = VK_ACCESS_TRANSFER_READ_BIT;

  vkCmdPipelineBarrier(cmdBuff,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT,
                       0,
                       0, nullptr,
                       1, &bufBarr,
                       0, nullptr);

  VkBufferCopy region0 = {};
  region0.srcOffset = 0;
  region0.dstOffset = 0;
  region0.size      = a_size;

  vkCmdCopyBuffer(cmdBuff, auxBuff, stagingBuff, 1, &region0);

  vkEndCommandBuffer(cmdBuff);

  vk_utils::executeCommandBufferNow(cmdBuff, queue, dev);

  // second, copy data from staging buff to a_dst
  //
  void* mappedMemory = nullptr;
  vkMapMemory(dev, stagingBuffMemory, 0, a_size, 0, &mappedMemory);
  memcpy(a_dst, mappedMemory, a_size);
  vkUnmapMemory(dev, stagingBuffMemory);
}
