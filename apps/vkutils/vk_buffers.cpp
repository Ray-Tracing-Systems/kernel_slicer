#include "vk_buffers.h"
#include "vk_utils.h"

namespace vk_utils
{
  VkBuffer createBuffer(VkDevice a_dev, VkDeviceSize a_size, VkBufferUsageFlags a_usageFlags, VkMemoryRequirements* a_pMemReq)
  {
    assert(a_dev != VK_NULL_HANDLE);
    
    VkBuffer result = VK_NULL_HANDLE;
    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size        = a_size;
    bufferCreateInfo.usage       = a_usageFlags;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateBuffer(a_dev, &bufferCreateInfo, VK_NULL_HANDLE, &result));
    if(a_pMemReq != nullptr) 
      vkGetBufferMemoryRequirements(a_dev, result, a_pMemReq);

    return result;
  }

  void createBufferStaging(VkDevice a_device, VkPhysicalDevice a_physDevice, const size_t a_bufferSize,
                           VkBuffer &a_buf, VkDeviceMemory& a_mem)
  {

    VkBufferCreateInfo bufferCreateInfo = {};
    bufferCreateInfo.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size        = a_bufferSize;
    bufferCreateInfo.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, nullptr, &a_buf));

    VkMemoryRequirements memoryRequirements;
    vkGetBufferMemoryRequirements(a_device, a_buf, &memoryRequirements);

    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize  = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits,
                                                            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                                                            a_physDevice);

    VK_CHECK_RESULT(vkAllocateMemory(a_device, &allocateInfo, nullptr, &a_mem));
    VK_CHECK_RESULT(vkBindBufferMemory(a_device, a_buf, a_mem, 0));
  }

  size_t getPaddedSize(size_t a_size, size_t a_alignment)
  {
    if (a_size % a_alignment == 0)
      return a_size;
    else
    {
      size_t sizeCut = a_size - (a_size % a_alignment);
      return sizeCut + a_alignment;
    }
  }

  uint32_t getSBTAlignedSize(uint32_t value, uint32_t alignment)
  {
    return (value + alignment - 1) & ~(alignment - 1);
  }


  std::vector<size_t> assignMemOffsetsWithPadding(const std::vector<VkMemoryRequirements> &a_memInfos)
  {
    assert(!a_memInfos.empty());

    std::vector<VkDeviceSize> mem_offsets;
    size_t currOffset = 0;
    for (size_t i = 0; i < a_memInfos.size() - 1; i++)
    {
      mem_offsets.push_back(currOffset);
      currOffset += vk_utils::getPaddedSize(a_memInfos[i].size, a_memInfos[i + 1].alignment);
    }

    auto last = a_memInfos.size() - 1;
    mem_offsets.push_back(currOffset);
    currOffset += vk_utils::getPaddedSize(a_memInfos[last].size, a_memInfos[last].alignment);

    // put total mem amount in last vector element
    mem_offsets.push_back(currOffset);

    return mem_offsets;
  }

  VkDeviceMemory allocateAndBindWithPadding(VkDevice a_dev, VkPhysicalDevice a_physDev, const std::vector<VkBuffer> &a_buffers,
                                            VkMemoryAllocateFlags flags)
  {
    if(a_buffers.empty())
    {
      logWarning("[allocateAndBindWithPadding]: buffers vector is empty");
      return VK_NULL_HANDLE;
    }

    std::vector<VkMemoryRequirements> memInfos(a_buffers.size());
    for(size_t i = 0; i < memInfos.size(); ++i)
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
        logWarning("[allocateAndBindWithPadding]: input buffers has different memReq.memoryTypeBits");
        return VK_NULL_HANDLE;
      }
    }

    auto offsets  = assignMemOffsetsWithPadding(memInfos);
    auto memTotal = offsets[offsets.size() - 1];

    VkDeviceMemory res;
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = memTotal;
    allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memInfos[0].memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDev);

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    if(flags)
    {
      memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
      memoryAllocateFlagsInfo.flags = flags;

      allocateInfo.pNext = &memoryAllocateFlagsInfo;
    }

    VK_CHECK_RESULT(vkAllocateMemory(a_dev, &allocateInfo, NULL, &res));

    for (size_t i = 0; i < memInfos.size(); i++)
    {
      if(a_buffers[i] != VK_NULL_HANDLE)
        vkBindBufferMemory(a_dev, a_buffers[i], res, offsets[i]);
    }

    return res;
  }

  std::vector<size_t> calculateMemOffsets(const std::vector<VkMemoryRequirements> &a_memReqs)
  {
    assert(!a_memReqs.empty());

    std::vector<VkDeviceSize> mem_offsets;
    size_t currOffset = 0;
    for (size_t i = 0; i < a_memReqs.size() - 1; i++)
    {
      mem_offsets.push_back(currOffset);
      currOffset += getPaddedSize(a_memReqs[i].size, a_memReqs[i + 1].alignment);
    }

    // put mem offset for last element of 'a_memInfos'
    //
    size_t last = a_memReqs.size() - 1;
    mem_offsets.push_back(currOffset);
    currOffset += getPaddedSize(a_memReqs[last].size, a_memReqs[last].alignment);

    // put total mem amount in last vector element
    //
    mem_offsets.push_back(currOffset);
    return mem_offsets;
  }

}

