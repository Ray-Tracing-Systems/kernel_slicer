#ifndef CHIMERA_VK_RT_UTILS_H
#define CHIMERA_VK_RT_UTILS_H

#include "vk_include.h"
#include "geom/vk_mesh.h"
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
    uint64_t deviceAddress = 0;
    VkBuffer buffer = VK_NULL_HANDLE;
    VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
  };

  struct ShaderBindingTable
  {
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkStridedDeviceAddressRegionKHR stridedDeviceAddress = {};
  };

  uint64_t getBufferDeviceAddress(VkDevice a_device, VkBuffer a_buffer);
  RTScratchBuffer allocScratchBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, VkDeviceSize size);

  AccelStructure createAccelStruct(VkDevice a_device, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo);
  VkDeviceMemory allocAndGetAddressAccelStructs(VkDevice a_device, VkPhysicalDevice a_physicalDevice, std::vector<AccelStructure> &a_as);
  VkDeviceMemory allocAndGetAddressAccelStruct(VkDevice a_device, VkPhysicalDevice a_physicalDevice, AccelStructure &a_as);

  VkStridedDeviceAddressRegionKHR getSBTStridedDeviceAddressRegion(VkDevice a_device, VkBuffer buffer,
                                                                   uint32_t handleCount, uint32_t handleSizeAligned);

  struct AccelStructureSizeInfo
  {
    size_t accelerationStructureSize;
    size_t updateScratchSize;
    size_t buildScratchSize;
  };

  AccelStructureSizeInfo getBuildBLASSizeInfo(VkDevice a_device, uint32_t maxVertexCount, uint32_t maxPrimitiveCount,
    size_t singleVertexSize);


  struct BLASBuildInput
  {
    std::vector<VkAccelerationStructureGeometryKHR> geom;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> buildRange;
    VkBuildAccelerationStructureFlagsKHR buildFlags = 0;
  };

  class AccelStructureBuilderV2
  {
  public:

    AccelStructureBuilderV2(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_queueIdx, VkQueue a_queue = VK_NULL_HANDLE);
    ~AccelStructureBuilderV2();

    // if (a_buildAsAdd == true) start building BLAS immediately after call to AddBLAS
    void Init(uint32_t maxVertexCountPerMesh, uint32_t maxPrimitiveCountPerMesh, uint32_t maxTotalPrimitiveCount,
        size_t singleVertexSize, bool a_buildAsAdd = false,
        VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    uint32_t AddBLAS(const MeshInfo &a_meshInfo, size_t a_vertexDataStride,
      VkDeviceOrHostAddressConstKHR a_vertexBufAddress, VkDeviceOrHostAddressConstKHR a_indexBufAddress,
      VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    void BuildAllBLAS();

    void UpdateBLAS(uint32_t idx, const MeshInfo &a_meshInfo,
      VkDeviceOrHostAddressConstKHR a_vertexBufAddress, VkDeviceOrHostAddressConstKHR a_indexBufAddress,
      VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    void BuildTLAS(uint32_t a_instNum, VkDeviceOrHostAddressConstKHR a_instBufAddress,
      VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      bool a_update = false);

    VkAccelerationStructureKHR GetTLAS() const { return m_tlas.handle; };
    VkAccelerationStructureKHR GetBLAS(uint32_t idx) const { assert(idx < m_blas.size()); return m_blas[idx].handle; };
    uint64_t GetBLASDeviceAddress(uint32_t idx) const { assert(idx < m_blas.size()); return m_blas[idx].deviceAddress; };

    void Destroy();

  private:
    bool m_queueBuild = false;
    VkAccelerationStructureBuildSizesInfoKHR GetSizeInfo(const VkAccelerationStructureBuildGeometryInfoKHR& a_buildInfo,
      std::vector<VkAccelerationStructureBuildRangeInfoKHR>& a_ranges);

    void BuildOneBLAS(uint32_t idx);
    void AllocBLASMemory(uint32_t a_memTypeIdx, VkDeviceSize a_size);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;

    VkQueue  m_queue    = VK_NULL_HANDLE;
    uint32_t m_queueIdx = UINT32_MAX;

    VkCommandPool m_cmdPool = VK_NULL_HANDLE;
    VkSemaphore m_buildSemaphore = VK_NULL_HANDLE;

    RTScratchBuffer m_scratchBuf = {};
    VkDeviceSize m_scratchSize   = 0;
    VkDeviceSize m_totalBLASSize = 0;
    VkDeviceSize m_totalBLASSizeEstimate = 0;

    VkDeviceSize m_currentBLASMemOffset = 0;
    VkDeviceSize m_lastAddedBLASSize    = 0;

    VkDeviceMemory m_blasMem = VK_NULL_HANDLE;
    VkDeviceMemory m_tlasMem = VK_NULL_HANDLE;

    std::vector<vk_rt_utils::AccelStructure> m_blas;
    std::vector<VkFence> m_buildFences;
    vk_rt_utils::AccelStructure m_tlas{};

    std::vector<BLASBuildInput> m_blasInputs;
  };

  class AccelStructureBuilder
  {
  public:

    AccelStructureBuilder(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_queueIdx, VkQueue a_queue = VK_NULL_HANDLE);
    ~AccelStructureBuilder();

    void BuildBLAS(std::vector<BLASBuildInput>& a_input,
      VkBuildAccelerationStructureFlagsKHR a_commonFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
//    void BuildTLAS(const std::vector<VkAccelerationStructureInstanceKHR>& a_instances,
//      VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
//      bool a_update = false);
    void BuildTLAS(uint32_t a_instNum, VkBuffer a_instBuffer, VkDeviceSize a_bufOffset = 0,
      VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
      bool a_update = false);
    void UpdateBLAS(uint32_t idx, BLASBuildInput & a_input,
      VkBuildAccelerationStructureFlagsKHR a_flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);

    VkAccelerationStructureKHR GetTLAS() const { return m_tlas.handle; };
    VkAccelerationStructureKHR GetBLAS(uint32_t idx) const { assert(idx < m_blas.size()); return m_blas[idx].handle; };
    uint64_t GetBLASDeviceAddress(uint32_t idx) const { assert(idx < m_blas.size()); return m_blas[idx].deviceAddress; };

    void Destroy();

  private:
    VkAccelerationStructureBuildSizesInfoKHR GetSizeInfo(const VkAccelerationStructureBuildGeometryInfoKHR& a_buildInfo,
      std::vector<VkAccelerationStructureBuildRangeInfoKHR>& a_ranges);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queueIdx = UINT32_MAX;
    VkCommandPool m_cmdPool = VK_NULL_HANDLE;

    VkDeviceMemory m_blasMem = VK_NULL_HANDLE;
    VkDeviceMemory m_tlasMem = VK_NULL_HANDLE;
    std::vector<BLASBuildInput> m_blasInputData;
    std::vector<vk_rt_utils::AccelStructure> m_blas;
    vk_rt_utils::AccelStructure m_tlas{};

    RTScratchBuffer m_scratchBuf = {};
    VkDeviceSize m_scratchSize   = 0;
    VkDeviceSize m_totalBLASSize = 0;
  };

  struct RTPipelineMaker
  {
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shaderGroups{};
    std::vector<VkShaderModule> shaderModules{};
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages;

    void             LoadShaders(VkDevice a_device, const std::vector<std::pair<VkShaderStageFlagBits, std::string>> &shader_paths);
    VkPipelineLayout MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout);
    VkPipelineLayout MakeLayout(VkDevice a_device, std::vector<VkDescriptorSetLayout> a_dslayouts);
    VkPipeline       MakePipeline(VkDevice a_device, uint32_t a_maxDepth = 2);

    private:
      int              m_stagesNum = 0;
      VkPipeline       m_pipeline  = VK_NULL_HANDLE;
      VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
  };

  VkTransformMatrixKHR transformMatrixFromRowMajArray(const std::array<float, 16> &m);
}


#endif//CHIMERA_VK_RT_UTILS_H
