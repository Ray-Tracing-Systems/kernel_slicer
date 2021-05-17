#ifndef CHIMERA_VK_RT_UTILS_H
#define CHIMERA_VK_RT_UTILS_H

#include "volk.h"

namespace vk_rt_utils
{
  struct RTScratchBuffer
  {
    uint64_t deviceAddress = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };

  struct AccelStructure
  {
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
    uint64_t deviceAddress = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
  };

  struct BLASInput
  {
    VkAccelerationStructureGeometryKHR geom;
    VkAccelerationStructureBuildRangeInfoKHR offsetInfo;
  };

struct ShaderBindingTable
{
  VkBuffer buf;
  VkDeviceSize size;
  VkStridedDeviceAddressRegionKHR stridedDeviceAddress;
};

  uint64_t getBufferDeviceAddress(VkDevice a_device, VkBuffer a_buffer);
  RTScratchBuffer allocScratchBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, VkDeviceSize size);

  void createAccelerationStructure(AccelStructure& accel, VkAccelerationStructureTypeKHR type,
                                   VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo,
                                   VkDevice a_device, VkPhysicalDevice a_physicalDevice);

  VkStridedDeviceAddressRegionKHR getSBTStridedDeviceAddressRegion(VkDevice a_device, VkBuffer buffer,
                                                                   uint32_t handleCount, uint32_t handleSizeAligned);
}


#endif//CHIMERA_VK_RT_UTILS_H
