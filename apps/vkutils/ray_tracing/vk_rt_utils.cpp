#include <vk_buffers.h>
#include "vk_rt_utils.h"
#include "vk_utils.h"
#include "vk_rt_funcs.h"

namespace vk_rt_utils
{

  VkTransformMatrixKHR transformMatrixFromRowMajArray(const std::array<float, 16> &m)
  {
    VkTransformMatrixKHR transformMatrix;
    for(int i = 0; i < 3; ++i)
    {
      for(int j = 0; j < 4; ++j)
      {
        transformMatrix.matrix[i][j] = m[i * 4 + j];
      }
    }

    return transformMatrix;
  }

  uint64_t getBufferDeviceAddress(VkDevice a_device, VkBuffer a_buffer)
  {
    VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = a_buffer;
    return vkGetBufferDeviceAddressKHR(a_device, &bufferDeviceAddressInfo);
  }

  RTScratchBuffer allocScratchBuffer(VkDevice a_device, VkPhysicalDevice a_physDevice, VkDeviceSize size)
  {
    RTScratchBuffer m_scratchBuf{};

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = size;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, nullptr, &m_scratchBuf.buffer));

    VkMemoryRequirements memoryRequirements{};
    vkGetBufferMemoryRequirements(a_device, m_scratchBuf.buffer, &memoryRequirements);

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    VkMemoryAllocateInfo memoryAllocateInfo = {};
    memoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memoryAllocateInfo.pNext = &memoryAllocateFlagsInfo;
    memoryAllocateInfo.allocationSize = memoryRequirements.size;
    memoryAllocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memoryRequirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, a_physDevice);

    VK_CHECK_RESULT(vkAllocateMemory(a_device, &memoryAllocateInfo, nullptr, &m_scratchBuf.memory));
    VK_CHECK_RESULT(vkBindBufferMemory(a_device, m_scratchBuf.buffer, m_scratchBuf.memory, 0));

    VkBufferDeviceAddressInfoKHR bufferDeviceAddressInfo{};
    bufferDeviceAddressInfo.sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddressInfo.buffer = m_scratchBuf.buffer;
    m_scratchBuf.deviceAddress    = vkGetBufferDeviceAddressKHR(a_device, &bufferDeviceAddressInfo);

    return m_scratchBuf;
  }

  VkDeviceMemory allocAndGetAddressAccelStruct(VkDevice a_device, VkPhysicalDevice a_physicalDevice, AccelStructure &a_as)
  {
    auto mem = vk_utils::allocateAndBindWithPadding(a_device, a_physicalDevice, {a_as.buffer}, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR);

    VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
    accelerationDeviceAddressInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    accelerationDeviceAddressInfo.accelerationStructure = a_as.handle;
    a_as.deviceAddress                                  = vkGetAccelerationStructureDeviceAddressKHR(a_device, &accelerationDeviceAddressInfo);

    return mem;
  }

  VkDeviceMemory allocAndGetAddressAccelStructs(VkDevice a_device, VkPhysicalDevice a_physicalDevice, std::vector<AccelStructure> &a_as)
  {
    std::vector<VkBuffer> bufs(a_as.size());
    for(size_t i = 0; i < bufs.size(); ++i)
      bufs[i] = a_as[i].buffer;

    auto mem = vk_utils::allocateAndBindWithPadding(a_device, a_physicalDevice, bufs, VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR);

    for(auto& as : a_as)
    {
      VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
      accelerationDeviceAddressInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      accelerationDeviceAddressInfo.accelerationStructure = as.handle;
      as.deviceAddress                                    = vkGetAccelerationStructureDeviceAddressKHR(a_device, &accelerationDeviceAddressInfo);
    }

    return mem;
  }

  AccelStructure createAccelStruct(VkDevice a_device, VkAccelerationStructureTypeKHR type, VkAccelerationStructureBuildSizesInfoKHR buildSizeInfo)
  {
    AccelStructure accel = {};

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size  = buildSizeInfo.accelerationStructureSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RESULT(vkCreateBuffer(a_device, &bufferCreateInfo, nullptr, &accel.buffer));

    // Acceleration structure
    VkAccelerationStructureCreateInfoKHR accelerationStructureCreate_info{};
    accelerationStructureCreate_info.sType  = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    accelerationStructureCreate_info.buffer = accel.buffer;
    accelerationStructureCreate_info.size = buildSizeInfo.accelerationStructureSize;
    accelerationStructureCreate_info.type = type;
    vkCreateAccelerationStructureKHR(a_device, &accelerationStructureCreate_info, nullptr, &accel.handle);

    return accel;
  }

  VkStridedDeviceAddressRegionKHR getSBTStridedDeviceAddressRegion(VkDevice a_device, VkBuffer buffer,
                                                                   uint32_t handleCount, uint32_t handleSizeAligned)
  {
    VkStridedDeviceAddressRegionKHR stridedDeviceAddressRegionKHR{};
    stridedDeviceAddressRegionKHR.deviceAddress = getBufferDeviceAddress(a_device, buffer);;
    stridedDeviceAddressRegionKHR.stride = handleSizeAligned;
    stridedDeviceAddressRegionKHR.size = handleCount * handleSizeAligned;

    return stridedDeviceAddressRegionKHR;
  }

  AccelStructureSizeInfo getBuildBLASSizeInfo(VkDevice a_device, uint32_t maxVertexCount, uint32_t maxPrimitiveCount,
    size_t singleVertexSize)
  {
    AccelStructureSizeInfo res = {};

    VkAccelerationStructureGeometryKHR accelerationStructureGeometry {};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
//    accelerationStructureGeometry.geometry.triangles.vertexData   = vertexBufAddress;
    accelerationStructureGeometry.geometry.triangles.maxVertex    = maxVertexCount;
    accelerationStructureGeometry.geometry.triangles.vertexStride = singleVertexSize;

    accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
//    accelerationStructureGeometry.geometry.triangles.indexData = indexBufAddress;
    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
    accelerationStructureGeometry.geometry.triangles.transformData.hostAddress   = nullptr;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags                    = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount            = 1;
    buildInfo.pGeometries              = &accelerationStructureGeometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(a_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &maxPrimitiveCount, &sizeInfo);

    res.accelerationStructureSize = sizeInfo.accelerationStructureSize;
    res.buildScratchSize  = sizeInfo.buildScratchSize;
    res.updateScratchSize = sizeInfo.updateScratchSize;

    return res;
  }


  ////////////////////////////////////////////////////////////////////////
  // AccelStructureBuilder

  AccelStructureBuilder::AccelStructureBuilder(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_queueIdx, VkQueue a_queue) : m_device(a_device),
                                                                                                                                         m_physDevice(a_physDevice),
                                                                                                                                         m_queueIdx(a_queueIdx)
  {
    if(a_queue != VK_NULL_HANDLE)
    {
      m_queue = a_queue;
    }
    else
    {
      vkGetDeviceQueue(m_device, m_queueIdx, 0, &m_queue);
    }

    m_cmdPool = vk_utils::createCommandPool(m_device, m_queueIdx, VkCommandPoolCreateFlagBits(0));
  }

  VkAccelerationStructureBuildSizesInfoKHR AccelStructureBuilder::GetSizeInfo(const VkAccelerationStructureBuildGeometryInfoKHR& a_buildInfo, std::vector<VkAccelerationStructureBuildRangeInfoKHR>& a_ranges)
  {
    std::vector<uint32_t> maxPrimCount(a_ranges.size());
    for(size_t i = 0; i < a_ranges.size(); ++i)
      maxPrimCount[i] = a_ranges[i].primitiveCount;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &a_buildInfo, maxPrimCount.data(), &sizeInfo);

    return sizeInfo;
  }

  void AccelStructureBuilder::BuildBLAS(std::vector<BLASBuildInput>& a_input, VkBuildAccelerationStructureFlagsKHR a_commonFlags)
  {
    m_blasInputData = a_input;

    auto nBlas = m_blasInputData.size();
    m_blas.resize(nBlas);

    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(nBlas);
    for(uint32_t idx = 0; idx < nBlas; idx++)
    {
      buildInfos[idx].sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
      buildInfos[idx].flags                    = a_commonFlags | m_blasInputData[idx].buildFlags;
      buildInfos[idx].geometryCount            = m_blasInputData[idx].geom.size();
      buildInfos[idx].pGeometries              = m_blasInputData[idx].geom.data();
      buildInfos[idx].mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
      buildInfos[idx].type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
      buildInfos[idx].srcAccelerationStructure = VK_NULL_HANDLE;

      auto sizeInfo = GetSizeInfo(buildInfos[idx], m_blasInputData[idx].buildRange);

      m_totalBLASSize += sizeInfo.accelerationStructureSize;
      m_scratchSize    = std::max(m_scratchSize, sizeInfo.buildScratchSize);

      m_blas[idx] = vk_rt_utils::createAccelStruct(m_device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo);
    }

    m_blasMem = vk_rt_utils::allocAndGetAddressAccelStructs(m_device, m_physDevice, m_blas);

    m_scratchBuf = vk_rt_utils::allocScratchBuffer(m_device, m_physDevice, m_scratchSize);

    std::vector<VkCommandBuffer> buildCmdBufs = vk_utils::createCommandBuffers(m_device, m_cmdPool, nBlas);
    for(uint32_t idx = 0; idx < nBlas; idx++)
    {
      VkCommandBufferBeginInfo cmdBufInfo = {};
      cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO ;
      VK_CHECK_RESULT(vkBeginCommandBuffer(buildCmdBufs[idx], &cmdBufInfo));
      auto& blas   = m_blas[idx];
      buildInfos[idx].dstAccelerationStructure  = blas.handle;
      buildInfos[idx].scratchData.deviceAddress = m_scratchBuf.deviceAddress;

      const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffset = m_blasInputData[idx].buildRange.data();
      vkCmdBuildAccelerationStructuresKHR(buildCmdBufs[idx], 1, &buildInfos[idx], &pBuildOffset);

      VkMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(buildCmdBufs[idx], VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                           0, 1, &barrier, 0, nullptr, 0, nullptr);
      vkEndCommandBuffer(buildCmdBufs[idx]);
    }

    vk_utils::executeCommandBufferNow(buildCmdBufs, m_queue, m_device);
    buildCmdBufs.clear();

    if (m_scratchBuf.memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_scratchBuf.memory, nullptr);
      m_scratchBuf.memory = VK_NULL_HANDLE;
    }
    if (m_scratchBuf.buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(m_device, m_scratchBuf.buffer, nullptr);
      m_scratchBuf.buffer = VK_NULL_HANDLE;
    }
  }

  void AccelStructureBuilder::BuildTLAS(uint32_t a_instNum, VkBuffer a_instBuffer, VkDeviceSize a_bufOffset, VkBuildAccelerationStructureFlagsKHR a_flags, bool a_update)
  {
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk = {};
    instancesVk.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesVk.arrayOfPointers = VK_FALSE;
    instancesVk.data.deviceAddress = vk_rt_utils::getBufferDeviceAddress(m_device, a_instBuffer); // + a_bufOffset ??

    VkAccelerationStructureGeometryKHR topASGeometry = {};
    topASGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags         = a_flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &topASGeometry;
    buildInfo.mode = a_update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &a_instNum, &sizeInfo);

    if(!a_update)
    {
      m_tlas    = vk_rt_utils::createAccelStruct(m_device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo);
      m_tlasMem = vk_rt_utils::allocAndGetAddressAccelStruct(m_device, m_physDevice, m_tlas);
    }

    m_scratchBuf = vk_rt_utils::allocScratchBuffer(m_device, m_physDevice, sizeInfo.buildScratchSize);

    buildInfo.srcAccelerationStructure  = a_update ? m_tlas.handle : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure  = m_tlas.handle;
    buildInfo.scratchData.deviceAddress = m_scratchBuf.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo = {};
    accelerationStructureBuildRangeInfo.primitiveCount  = a_instNum;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.firstVertex     = 0;
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(m_device, m_cmdPool);
    VkCommandBufferBeginInfo cmdBufInfo = {};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));
    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, accelerationBuildStructureRangeInfos.data());
    vkEndCommandBuffer(commandBuffer);
    vk_utils::executeCommandBufferNow(commandBuffer, m_queue, m_device);

    if (m_scratchBuf.memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_scratchBuf.memory, nullptr);
      m_scratchBuf.memory = VK_NULL_HANDLE;
    }
    if (m_scratchBuf.buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(m_device, m_scratchBuf.buffer, nullptr);
      m_scratchBuf.buffer = VK_NULL_HANDLE;
    }
  }

  void AccelStructureBuilder::UpdateBLAS(uint32_t idx, BLASBuildInput & a_input, VkBuildAccelerationStructureFlagsKHR a_flags)
  {
    assert(size_t(idx) < m_blas.size());

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo= {};
    buildInfo.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags                    = a_flags;
    buildInfo.geometryCount            = (uint32_t)a_input.geom.size();
    buildInfo.pGeometries              = a_input.geom.data();
    buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
    buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = m_blas[idx].handle;
    buildInfo.dstAccelerationStructure = m_blas[idx].handle;

    auto sizeInfo = GetSizeInfo(buildInfo, m_blasInputData[idx].buildRange);

    m_scratchBuf = vk_rt_utils::allocScratchBuffer(m_device, m_physDevice, sizeInfo.buildScratchSize);

    const VkAccelerationStructureBuildRangeInfoKHR* pBuildOffset = a_input.buildRange.data();

    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(m_device, m_cmdPool);
    VkCommandBufferBeginInfo cmdBufInfo = {};
    cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

    vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, &pBuildOffset);

    vkEndCommandBuffer(commandBuffer);
    vk_utils::executeCommandBufferNow(commandBuffer, m_queue, m_device);

    if (m_scratchBuf.memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_scratchBuf.memory, nullptr);
      m_scratchBuf.memory = VK_NULL_HANDLE;
    }
    if (m_scratchBuf.buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(m_device, m_scratchBuf.buffer, nullptr);
      m_scratchBuf.buffer = VK_NULL_HANDLE;
    }
  }

  AccelStructureBuilder::~AccelStructureBuilder()
  {
    if(m_cmdPool != VK_NULL_HANDLE)
    {
      vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
    }

    Destroy();
  }

  void AccelStructureBuilder::Destroy()
  {
    for(auto& blas : m_blas)
    {
      if(blas.buffer != VK_NULL_HANDLE)
      {
        vkDestroyBuffer(m_device, blas.buffer, nullptr);
        blas.buffer = VK_NULL_HANDLE;
      }
    }
    if(m_blasMem != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_blasMem, nullptr);
      m_blasMem = VK_NULL_HANDLE;
    }

    if(m_tlas.buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(m_device, m_tlas.buffer, nullptr);
      m_tlas.buffer = VK_NULL_HANDLE;
    }

    if(m_tlasMem != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_tlasMem, nullptr);
      m_tlasMem = VK_NULL_HANDLE;
    }

    m_blasInputData.clear();
  }

  ////////////////////////////////////////////////////////////////////////
  // AccelStructureBuilderV2

  AccelStructureBuilderV2::AccelStructureBuilderV2(VkDevice a_device, VkPhysicalDevice a_physDevice,
    uint32_t a_queueIdx, VkQueue a_queue): m_device(a_device), m_physDevice(a_physDevice), m_queueIdx(a_queueIdx)
  {
    if(a_queue != VK_NULL_HANDLE)
    {
      m_queue = a_queue;
    }
    else
    {
      vkGetDeviceQueue(m_device, m_queueIdx, 0, &m_queue);
    }

    m_cmdPool = vk_utils::createCommandPool(m_device, m_queueIdx, VkCommandPoolCreateFlagBits(0));
  }

  AccelStructureBuilderV2::~AccelStructureBuilderV2()
  {
    if(m_cmdPool != VK_NULL_HANDLE)
    {
      vkDestroyCommandPool(m_device, m_cmdPool, nullptr);
    }

    if(m_buildSemaphore != VK_NULL_HANDLE)
    {
      vkDestroySemaphore(m_device, m_buildSemaphore, nullptr);
    }

    for(auto& fence : m_buildFences)
    {
      vkDestroyFence(m_device, fence, nullptr);
    }

    Destroy();
  }

  void AccelStructureBuilderV2::Init(uint32_t maxVertexCountPerMesh, uint32_t maxPrimitiveCountPerMesh, uint32_t maxTotalPrimitiveCount,
    size_t singleVertexSize, bool a_buildAsAdd, VkBuildAccelerationStructureFlagsKHR a_flags)
  {
    m_queueBuild = a_buildAsAdd;
    VkAccelerationStructureGeometryKHR accelerationStructureGeometry {};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    //    accelerationStructureGeometry.geometry.triangles.vertexData   = vertexBufAddress;
    accelerationStructureGeometry.geometry.triangles.maxVertex    = maxVertexCountPerMesh;
    accelerationStructureGeometry.geometry.triangles.vertexStride = singleVertexSize;

    accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
    //    accelerationStructureGeometry.geometry.triangles.indexData = indexBufAddress;
    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
    accelerationStructureGeometry.geometry.triangles.transformData.hostAddress   = nullptr;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType                    = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags                    = a_flags;
    buildInfo.mode                     = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type                     = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;
    buildInfo.geometryCount            = 1;
    buildInfo.pGeometries              = &accelerationStructureGeometry;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &maxPrimitiveCountPerMesh, &sizeInfo);

    m_scratchSize = sizeInfo.buildScratchSize;
    m_scratchBuf = vk_rt_utils::allocScratchBuffer(m_device, m_physDevice, m_scratchSize);

    m_totalBLASSizeEstimate = sizeInfo.accelerationStructureSize * (1 + maxTotalPrimitiveCount / maxPrimitiveCountPerMesh);
  }

  VkAccelerationStructureBuildSizesInfoKHR AccelStructureBuilderV2::GetSizeInfo(const VkAccelerationStructureBuildGeometryInfoKHR& a_buildInfo, std::vector<VkAccelerationStructureBuildRangeInfoKHR>& a_ranges)
  {
    std::vector<uint32_t> maxPrimCount(a_ranges.size());
    for(size_t i = 0; i < a_ranges.size(); ++i)
      maxPrimCount[i] = a_ranges[i].primitiveCount;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &a_buildInfo, maxPrimCount.data(), &sizeInfo);

    return sizeInfo;
  }

  uint32_t AccelStructureBuilderV2::AddBLAS(const MeshInfo &a_meshInfo, size_t a_vertexDataStride,
    VkDeviceOrHostAddressConstKHR a_vertexBufAddress, VkDeviceOrHostAddressConstKHR a_indexBufAddress,
    VkBuildAccelerationStructureFlagsKHR a_flags)
  {
    VkAccelerationStructureGeometryKHR accelerationStructureGeometry {};
    accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    accelerationStructureGeometry.geometry.triangles.vertexData   = a_vertexBufAddress;
    accelerationStructureGeometry.geometry.triangles.maxVertex    = a_meshInfo.m_vertNum;
    accelerationStructureGeometry.geometry.triangles.vertexStride = a_vertexDataStride;
    accelerationStructureGeometry.geometry.triangles.indexType    = VK_INDEX_TYPE_UINT32;
    accelerationStructureGeometry.geometry.triangles.indexData    = a_indexBufAddress;
    accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
    accelerationStructureGeometry.geometry.triangles.transformData.hostAddress   = nullptr;

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
    accelerationStructureBuildRangeInfo.primitiveCount  = a_meshInfo.m_indNum / 3;
    accelerationStructureBuildRangeInfo.primitiveOffset = a_meshInfo.m_indexBufOffset;
    accelerationStructureBuildRangeInfo.firstVertex     = a_meshInfo.m_vertexOffset;
    accelerationStructureBuildRangeInfo.transformOffset = 0;

    vk_rt_utils::BLASBuildInput blasInput;
    blasInput.geom.emplace_back(accelerationStructureGeometry);
    blasInput.buildRange.emplace_back(accelerationStructureBuildRangeInfo);
    m_blasInputs.emplace_back(blasInput);

    uint32_t idx = m_blasInputs.size() - 1;

    if(m_queueBuild)
    {
      BuildOneBLAS(idx);
    }

    return idx;
  }


  void AccelStructureBuilderV2::AllocBLASMemory(uint32_t a_memTypeIdx, VkDeviceSize a_size)
  {
    VkMemoryAllocateInfo allocateInfo = {};
    allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.pNext           = nullptr;
    allocateInfo.allocationSize  = a_size;
    allocateInfo.memoryTypeIndex = a_memTypeIdx;

    VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
    memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

    allocateInfo.pNext = &memoryAllocateFlagsInfo;

    VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, VK_NULL_HANDLE, &m_blasMem));
  }

  void AccelStructureBuilderV2::BuildOneBLAS(uint32_t idx)
  {
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType                                       = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags                                       = m_blasInputs[idx].buildFlags;
    buildInfo.geometryCount                               = m_blasInputs[idx].geom.size();
    buildInfo.pGeometries                                 = m_blasInputs[idx].geom.data();
    buildInfo.mode                                        = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type                                        = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.srcAccelerationStructure                    = VK_NULL_HANDLE;

    auto sizeInfo = GetSizeInfo(buildInfo, m_blasInputs[idx].buildRange);

    m_totalBLASSize += sizeInfo.accelerationStructureSize;
    if(m_totalBLASSize > m_totalBLASSizeEstimate)
    {
      std::stringstream ss;
      ss << "[AccelStructureBuilderV2::BuildOneBLAS(" << idx << ")]: estimated total BLAS memory size (" << m_totalBLASSizeEstimate
         << ") exceeded (" << m_totalBLASSize << "). Need realloc (not implemented yet). Shutting down.";
      vk_utils::logWarning(ss.str());
      //      assert(m_totalBLASSize >= sizeInfo.buildScratchSize);
      RUN_TIME_ERROR(ss.str().c_str());
    }

    if(m_totalBLASSize > m_totalBLASSizeEstimate)
    {
      std::stringstream ss;
      ss << "[AccelStructureBuilderV2::BuildOneBLAS(" << idx << ")]: estimated scratch buffer size (" << m_scratchSize
         << ") exceeded (" << sizeInfo.buildScratchSize << "). Need realloc (not implemented yet). Shutting down.";
      vk_utils::logWarning(ss.str());
      //      assert(m_scratchSize >= sizeInfo.buildScratchSize);
      RUN_TIME_ERROR(ss.str().c_str());
    }

    m_blas.push_back(vk_rt_utils::createAccelStruct(m_device, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizeInfo));

    VkMemoryRequirements memInfo = {};
    vkGetBufferMemoryRequirements(m_device, m_blas[idx].buffer, &memInfo);

    if(m_blasMem == VK_NULL_HANDLE)
    {
      auto memType = vk_utils::findMemoryType(memInfo.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_physDevice);
      AllocBLASMemory(memType, m_totalBLASSizeEstimate);
      m_currentBLASMemOffset = 0;
    }
    else
    {
      m_currentBLASMemOffset += vk_utils::getPaddedSize(m_lastAddedBLASSize, memInfo.alignment);
    }
    m_lastAddedBLASSize = memInfo.size;
    vkBindBufferMemory(m_device, m_blas[idx].buffer, m_blasMem, m_currentBLASMemOffset);

    {
      VkAccelerationStructureDeviceAddressInfoKHR accelerationDeviceAddressInfo{};
      accelerationDeviceAddressInfo.sType                 = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
      accelerationDeviceAddressInfo.accelerationStructure = m_blas[idx].handle;
      m_blas[idx].deviceAddress                           = vkGetAccelerationStructureDeviceAddressKHR(m_device, &accelerationDeviceAddressInfo);
    }

    buildInfo.dstAccelerationStructure  = m_blas[idx].handle;
    buildInfo.scratchData.deviceAddress = m_scratchBuf.deviceAddress;

    {
      VkFence fence;
      VkFenceCreateInfo fenceCreateInfo = {};
      fenceCreateInfo.sType             = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fenceCreateInfo.flags             = 0;
      VK_CHECK_RESULT(vkCreateFence(m_device, &fenceCreateInfo, NULL, &fence));
      m_buildFences.push_back(fence);
    }

    VkCommandBuffer buildCmdBuf = vk_utils::createCommandBuffer(m_device, m_cmdPool);
    {
      VkCommandBufferBeginInfo cmdBufInfo = {};
      cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO ;
      cmdBufInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      VK_CHECK_RESULT(vkBeginCommandBuffer(buildCmdBuf, &cmdBufInfo));

      const VkAccelerationStructureBuildRangeInfoKHR *pBuildOffset = m_blasInputs[idx].buildRange.data();
      vkCmdBuildAccelerationStructuresKHR(buildCmdBuf, 1, &buildInfo, &pBuildOffset);

      VkMemoryBarrier barrier = {};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
      barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
      vkCmdPipelineBarrier(buildCmdBuf,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        0, 1, &barrier, 0, nullptr, 0, nullptr);

      vkEndCommandBuffer(buildCmdBuf);
    }

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers    = &buildCmdBuf;

//    std::cout << "Submitting BLAS#" << idx << " for building...\n";
    VK_CHECK_RESULT(vkQueueSubmit(m_queue, 1, &submitInfo, m_buildFences.back()));
  }

  // @TODO
  void AccelStructureBuilderV2::UpdateBLAS(uint32_t idx, const MeshInfo &a_meshInfo,
    VkDeviceOrHostAddressConstKHR a_vertexBufAddress, VkDeviceOrHostAddressConstKHR a_indexBufAddress,
    VkBuildAccelerationStructureFlagsKHR a_flags)
  {
  }

  void AccelStructureBuilderV2::BuildAllBLAS()
  {
    if(!m_queueBuild)
    {
      for(size_t i = 0; i < m_blasInputs.size(); ++i)
        BuildOneBLAS(i);
    }
    VK_CHECK_RESULT(vkWaitForFences(m_device, m_buildFences.size(), m_buildFences.data(), VK_TRUE, vk_utils::DEFAULT_TIMEOUT));
  }

  void AccelStructureBuilderV2::BuildTLAS(uint32_t a_instNum, VkDeviceOrHostAddressConstKHR a_instBufAddress,
    VkBuildAccelerationStructureFlagsKHR a_flags, bool a_update)
  {
    VkAccelerationStructureGeometryInstancesDataKHR instancesVk = {};
    instancesVk.sType              = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    instancesVk.arrayOfPointers    = VK_FALSE;
    instancesVk.data.deviceAddress = a_instBufAddress.deviceAddress;

    VkAccelerationStructureGeometryKHR topASGeometry = {};
    topASGeometry.sType        =VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    topASGeometry.geometry.instances = instancesVk;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo = {};
    buildInfo.sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.flags         = a_flags;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries   = &topASGeometry;
    buildInfo.mode = a_update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo = {};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &a_instNum, &sizeInfo);

    assert(sizeInfo.buildScratchSize <= m_scratchSize);

    if(!a_update)
    {
      m_tlas    = vk_rt_utils::createAccelStruct(m_device, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizeInfo);
      m_tlasMem = vk_rt_utils::allocAndGetAddressAccelStruct(m_device, m_physDevice, m_tlas);
    }

    buildInfo.srcAccelerationStructure  = a_update ? m_tlas.handle : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure  = m_tlas.handle;
    buildInfo.scratchData.deviceAddress = m_scratchBuf.deviceAddress;

    VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo = {};
    accelerationStructureBuildRangeInfo.primitiveCount  = a_instNum;
    accelerationStructureBuildRangeInfo.primitiveOffset = 0;
    accelerationStructureBuildRangeInfo.firstVertex     = 0;
    accelerationStructureBuildRangeInfo.transformOffset = 0;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> accelerationBuildStructureRangeInfos = { &accelerationStructureBuildRangeInfo };

    VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(m_device, m_cmdPool);
    {
      VkCommandBufferBeginInfo cmdBufInfo = {};
      cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO ;
      VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));
      vkCmdBuildAccelerationStructuresKHR(commandBuffer, 1, &buildInfo, accelerationBuildStructureRangeInfos.data());
      vkEndCommandBuffer(commandBuffer);
    }
    vk_utils::executeCommandBufferNow(commandBuffer, m_queue, m_device);
  }

  void AccelStructureBuilderV2::Destroy()
  {
    for(auto& blas : m_blas)
    {
      if(blas.handle != VK_NULL_HANDLE)
      {
        vkDestroyAccelerationStructureKHR(m_device, blas.handle, nullptr);
        blas.handle = VK_NULL_HANDLE;
      }

      if(blas.buffer != VK_NULL_HANDLE)
      {
        vkDestroyBuffer(m_device, blas.buffer, nullptr);
        blas.buffer = VK_NULL_HANDLE;
      }
    }
    if(m_blasMem != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_blasMem, nullptr);
      m_blasMem = VK_NULL_HANDLE;
    }

    if(m_tlas.buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(m_device, m_tlas.buffer, nullptr);
      m_tlas.buffer = VK_NULL_HANDLE;
    }

    if(m_tlas.handle != VK_NULL_HANDLE)
    {
      vkDestroyAccelerationStructureKHR(m_device, m_tlas.handle, nullptr);
      m_tlas.handle = VK_NULL_HANDLE;
    }

    if(m_tlasMem != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_tlasMem, nullptr);
      m_tlasMem = VK_NULL_HANDLE;
    }

    if(m_scratchBuf.buffer != VK_NULL_HANDLE)
    {
      vkDestroyBuffer(m_device, m_scratchBuf.buffer, nullptr);
      m_scratchBuf.buffer = VK_NULL_HANDLE;
    }

    if(m_scratchBuf.memory != VK_NULL_HANDLE)
    {
      vkFreeMemory(m_device, m_scratchBuf.memory, nullptr);
      m_scratchBuf.memory = VK_NULL_HANDLE;
    }

    m_currentBLASMemOffset = 0;
  }

  ////////////////////////////////////////////////////////////////////////
  // RTPipelineMaker

  VkPipelineLayout RTPipelineMaker::MakeLayout(VkDevice a_device, VkDescriptorSetLayout a_dslayout)
  {
    VkPipelineLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.pSetLayouts    = &a_dslayout;
    layoutCreateInfo.setLayoutCount = 1;
    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &layoutCreateInfo, nullptr, &m_pipelineLayout));

    return m_pipelineLayout;
  }

  VkPipelineLayout RTPipelineMaker::MakeLayout(VkDevice a_device, std::vector<VkDescriptorSetLayout> a_dslayouts)
  {
    VkPipelineLayoutCreateInfo layoutCreateInfo = {};
    layoutCreateInfo.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutCreateInfo.pSetLayouts    = a_dslayouts.data();
    layoutCreateInfo.setLayoutCount = a_dslayouts.size();
    VK_CHECK_RESULT(vkCreatePipelineLayout(a_device, &layoutCreateInfo, nullptr, &m_pipelineLayout));

    return m_pipelineLayout;
  }

  void RTPipelineMaker::LoadShaders(VkDevice a_device, const std::vector<std::pair<VkShaderStageFlagBits, std::string>> &shader_paths)
  {
    for(auto& [stage, path] : shader_paths)
    {
      VkPipelineShaderStageCreateInfo shaderStage = {};
      shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      shaderStage.stage = stage;

      auto shaderCode = vk_utils::readSPVFile(path.c_str());
      shaderModules.push_back(vk_utils::createShaderModule(a_device, shaderCode));
      shaderStage.module = shaderModules.back();

      shaderStage.pName = "main";
      assert(shaderStage.module != VK_NULL_HANDLE);
      shaderStages.push_back(shaderStage);

      VkRayTracingShaderGroupCreateInfoKHR shaderGroup{};
      shaderGroup.sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;

      if(stage == VK_SHADER_STAGE_MISS_BIT_KHR || stage == VK_SHADER_STAGE_RAYGEN_BIT_KHR ||
         stage == VK_SHADER_STAGE_CALLABLE_BIT_KHR)
      {
        shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
        shaderGroup.generalShader = static_cast<uint32_t>(shaderStages.size()) - 1;
        shaderGroup.closestHitShader = VK_SHADER_UNUSED_KHR;
      }
      else if(stage == VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR)
      {
        shaderGroup.type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
        shaderGroup.generalShader = VK_SHADER_UNUSED_KHR;
        shaderGroup.closestHitShader = static_cast<uint32_t>(shaderStages.size()) - 1;
      }
      // @TODO: intersection, procedural, anyhit
      shaderGroup.anyHitShader = VK_SHADER_UNUSED_KHR;
      shaderGroup.intersectionShader = VK_SHADER_UNUSED_KHR;

      shaderGroups.push_back(shaderGroup);
    }
  }

  VkPipeline RTPipelineMaker::MakePipeline(VkDevice a_device, uint32_t a_maxDepth)
  {
    VkRayTracingPipelineCreateInfoKHR createInfo = {};
    createInfo.sType      = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    createInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    createInfo.pStages    = shaderStages.data();
    createInfo.groupCount = static_cast<uint32_t>(shaderGroups.size());
    createInfo.pGroups    = shaderGroups.data();
    createInfo.maxPipelineRayRecursionDepth = a_maxDepth;
    createInfo.layout = m_pipelineLayout;
    VK_CHECK_RESULT(vkCreateRayTracingPipelinesKHR(a_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &createInfo, nullptr, &m_pipeline));

    return m_pipeline;
  }

}