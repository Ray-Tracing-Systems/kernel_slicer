#ifndef CHIMERA_VK_RT_UTILS_H
#define CHIMERA_VK_RT_UTILS_H

#define USE_VOLK
#include "vk_include.h"
#include <array>
#include <vector>
#include <string>

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

  struct RTPipelineMaker
  {
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};
    std::vector<VkShaderModule> shaderModules{};
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    void             LoadShaders(VkDevice a_device, const std::vector<std::pair<VkShaderStageFlagBits, std::string>> &shader_paths);
    VkPipelineLayout MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout);
    VkPipelineLayout MakeLayout(VkDevice a_device, std::vector<VkDescriptorSetLayout> a_dslayouts);
    VkPipeline       MakePipeline(VkDevice a_device);

    private:
      int              m_stagesNum = 0;
      VkPipeline       m_pipeline  = VK_NULL_HANDLE;
      VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
  };

  VkTransformMatrixKHR transformMatrixFromRowMajArray(const std::array<float, 16> &m);
}


#endif//CHIMERA_VK_RT_UTILS_H
