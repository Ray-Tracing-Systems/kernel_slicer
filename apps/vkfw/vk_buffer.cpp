#include <cassert>
#include <iostream>

#include "vk_utils.h"
#include "vk_buffer.h"

#if defined(__ANDROID__) // Dynamic load, use vulkan_wrapper.h to load vulkan functions
  #include "vulkan_wrapper/vulkan_wrapper.h"
#else
  #include <vulkan/vulkan.h>
#endif


VkBuffer vkfw::CreateBuffer(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags)
{
  VkBuffer buf = VK_NULL_HANDLE;
  VkBufferCreateInfo bufferCreateInfo = {};
  bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size        = a_size;
  bufferCreateInfo.usage       = a_usageFlags;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK_RESULT(vkCreateBuffer(a_dev, &bufferCreateInfo, VK_NULL_HANDLE, &buf));
  return buf;
}

VkMemoryRequirements vkfw::CreateBuffer(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, VkBuffer &a_buf)
{
  assert(a_dev != VK_NULL_HANDLE);
  if (a_buf != VK_NULL_HANDLE)
    vkDestroyBuffer(a_dev, a_buf, VK_NULL_HANDLE);
  VkBufferCreateInfo bufferCreateInfo = {};
  bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size        = a_size;
  bufferCreateInfo.usage       = a_usageFlags;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK_RESULT(vkCreateBuffer(a_dev, &bufferCreateInfo, VK_NULL_HANDLE, &a_buf));
  VkMemoryRequirements result;
  vkGetBufferMemoryRequirements(a_dev, a_buf, &result);
  return result;
}

vkfw::BufferReqPair vkfw::CreateBuffer2(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, VkBuffer& a_buf, const VkAllocationCallbacks* pAllocator)
{
  assert(a_dev != VK_NULL_HANDLE);
  if (a_buf != VK_NULL_HANDLE)
    vkDestroyBuffer(a_dev, a_buf, pAllocator);

  VkBufferCreateInfo bufferCreateInfo = {};
  bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size        = a_size;
  bufferCreateInfo.usage       = a_usageFlags;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK_RESULT(vkCreateBuffer(a_dev, &bufferCreateInfo, pAllocator, &a_buf));

  BufferReqPair res; 
  vkGetBufferMemoryRequirements(a_dev, a_buf, &res.req);
  res.buffer = a_buf;
  return res;
}

vkfw::BufferReqPair vkfw::CreateBuffer2(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, const VkAllocationCallbacks* pAllocator)
{
  assert(a_dev != VK_NULL_HANDLE);
  vkfw::BufferReqPair res; 

  VkBufferCreateInfo bufferCreateInfo = {};
  bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size        = a_size;
  bufferCreateInfo.usage       = a_usageFlags;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VK_CHECK_RESULT(vkCreateBuffer(a_dev, &bufferCreateInfo, pAllocator, &res.buffer));
  vkGetBufferMemoryRequirements(a_dev, res.buffer, &res.req);
  return res;
}


std::vector<size_t> vkfw::AssignMemOffsetsWithPadding(const std::vector<vkfw::BufferReqPair> a_memInfos)
{
  assert(a_memInfos.size() >= 1);
  
  std::vector<VkDeviceSize> mem_offsets;
  size_t currOffset = 0;
  for (size_t i = 0; i < a_memInfos.size() - 1; i++)
  {
    mem_offsets.push_back(currOffset);
    currOffset += vk_utils::Padding(a_memInfos[i].req.size, a_memInfos[i + 1].req.alignment);
  }

  // put mem offset for last element of 'a_memInfos'
  // 
  size_t last = a_memInfos.size() - 1;
  mem_offsets.push_back(currOffset);
  currOffset += vk_utils::Padding(a_memInfos[last].req.size, a_memInfos[last].req.alignment);
  
  // put total mem amount in last vector element
  //
  mem_offsets.push_back(currOffset);
  return mem_offsets;
}

std::vector<size_t> vkfw::AssignMemOffsetsWithPadding(const std::vector<VkMemoryRequirements> a_memInfos)
{
  assert(a_memInfos.size() >= 1);
 
  std::vector<VkDeviceSize> mem_offsets;
  size_t currOffset = 0;
  for (size_t i = 0; i < a_memInfos.size() - 1; i++)
  {
    mem_offsets.push_back(currOffset);
    currOffset += vk_utils::Padding(a_memInfos[i].size, a_memInfos[i + 1].alignment);
  }

  // put mem offset for last element of 'a_memInfos'
  // 
  size_t last = a_memInfos.size() - 1;
  mem_offsets.push_back(currOffset);
  currOffset += vk_utils::Padding(a_memInfos[last].size, a_memInfos[last].alignment);
  
  // put total mem amount in last vector element
  //
  mem_offsets.push_back(currOffset);
  return mem_offsets;
}

VkDeviceMemory vkfw::AllocateAndBindWithPadding(VkDevice a_dev, VkPhysicalDevice a_physDev, const std::vector<VkBuffer> a_buffers)
{
  if(a_buffers.size() == 0)
  {
    std::cout << "[vkfw::AllocateAndBindWithPadding]: error, zero input array" << std::endl;
    return VK_NULL_HANDLE;
  }

  std::vector<VkMemoryRequirements> memInfos(a_buffers.size());
  for(size_t i=0;i<memInfos.size();i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkGetBufferMemoryRequirements(a_dev, a_buffers[i], &memInfos[i]);
    else
    {
      memInfos[i] = memInfos[0];
      memInfos[i].size = 0;
    }
  }
  for(size_t i=1;i<memInfos.size();i++)
  {
    if(memInfos[i].memoryTypeBits != memInfos[0].memoryTypeBits)
    {
      std::cout << "[vkfw::AllocateAndBindWithPadding]: error, input buffers has different 'memReq.memoryTypeBits'" << std::endl;
      return VK_NULL_HANDLE;
    }
  }

  auto offsets  = AssignMemOffsetsWithPadding(memInfos);
  auto memTotal = offsets[offsets.size() - 1];

  VkDeviceMemory res;

  VkMemoryAllocateInfo allocateInfo = {};
  allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.pNext           = nullptr;
  allocateInfo.allocationSize  = memTotal;
  allocateInfo.memoryTypeIndex = vk_utils::FindMemoryType(memInfos[0].memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDev);
  VK_CHECK_RESULT(vkAllocateMemory(a_dev, &allocateInfo, NULL, &res));

  for (size_t i = 0; i < memInfos.size(); i++)
  {
    if(a_buffers[i] != VK_NULL_HANDLE)
      vkBindBufferMemory(a_dev, a_buffers[i], res, offsets[i]);
  }

  return res;
}
