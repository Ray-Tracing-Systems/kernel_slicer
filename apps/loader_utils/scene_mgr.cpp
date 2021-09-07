#include <map>
#include <array>
#include "scene_mgr.h"
#include "vk_utils.h"
#include "vk_buffers.h"
#include "../loader_utils/hydraxml.h"
#include <ray_tracing/vk_rt_funcs.h>

VkTransformMatrixKHR transformMatrixFromFloat4x4(const LiteMath::float4x4 &m)
{
  VkTransformMatrixKHR transformMatrix;
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 4; ++j)
    {
      transformMatrix.matrix[i][j] = m(i, j);
    }
  }

  return transformMatrix;
}

SceneManager::SceneManager(VkDevice a_device, VkPhysicalDevice a_physDevice,
                           uint32_t a_transferQId, uint32_t a_graphicsQId, bool useRTX, bool debug) : m_device(
    a_device), m_physDevice(a_physDevice),
                                                                                                      m_transferQId(
                                                                                                          a_transferQId),
                                                                                                      m_graphicsQId(
                                                                                                          a_graphicsQId),
                                                                                                      m_useRTX(useRTX),
                                                                                                      m_debug(debug)
{
  vkGetDeviceQueue(m_device, m_transferQId, 0, &m_transferQ);
  vkGetDeviceQueue(m_device, m_graphicsQId, 0, &m_graphicsQ);
  VkDeviceSize scratchMemSize = 64 * 1024 * 1024;
  m_pCopyHelper = std::make_unique<vk_utils::PingPongCopyHelper>(m_physDevice, m_device, m_transferQ, m_transferQId,
                                                                 scratchMemSize);
  m_pMeshData = std::make_shared<Mesh8F>();

}

bool SceneManager::LoadSingleMesh(const std::string &meshPath)
{
  auto meshId = AddMeshFromFile(meshPath);

  InstanceMesh(meshId, LiteMath::float4x4());
  LoadGeoDataOnGPU();
  if(m_useRTX)
    AddBLAS(meshId);

  return true;
}

bool SceneManager::LoadSceneXML(const std::string &scenePath, bool transpose)
{
  auto hscene_main = std::make_shared<hydra_xml::HydraScene>();
  auto res = hscene_main->LoadState(scenePath);

  if (res < 0)
  {
    RUN_TIME_ERROR("LoadSceneXML error");
    return false;
  }

  for (size_t i = 0; i < hscene_main->m_meshloc.size(); ++i)
  {
    auto meshId = AddMeshFromFile(hscene_main->m_meshloc[i]);

    auto instances = hscene_main->m_instancesPerMeshLoc[hscene_main->m_meshloc[i]];
    for (size_t j = 0; j < instances.size(); ++j)
    {
      if (transpose)
        InstanceMesh(meshId, LiteMath::transpose(instances[j]));
      else
        InstanceMesh(meshId, instances[j]);
    }
  }

  LoadGeoDataOnGPU();
  hscene_main = nullptr;

  if (m_useRTX)
  {
    for (size_t i = 0; i < m_meshInfos.size(); ++i)
    {
      AddBLAS(i);
    }
  }

  return true;
}

void SceneManager::LoadSingleTriangle()
{
  std::vector<Vertex> vertices =
      {
          {{1.0f,  1.0f,  0.0f}},
          {{-1.0f, 1.0f,  0.0f}},
          {{0.0f,  -1.0f, 0.0f}}
      };

  std::vector<uint32_t> indices = {0, 1, 2};
  m_totalIndices = static_cast<uint32_t>(indices.size());

  VkDeviceSize vertexBufSize = sizeof(Vertex) * vertices.size();
  VkDeviceSize indexBufSize = sizeof(uint32_t) * indices.size();

  VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (m_useRTX)
  {
    flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }

  const VkBufferUsageFlags vertFlags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags;
  m_geoVertBuf = vk_utils::createBuffer(m_device, vertexBufSize, vertFlags);

  const VkBufferUsageFlags idxFlags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | flags;
  m_geoIdxBuf = vk_utils::createBuffer(m_device, indexBufSize, idxFlags);

  VkMemoryAllocateFlags allocFlags{};
  if (m_useRTX)
  {
    allocFlags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
  }

  m_geoMemAlloc = vk_utils::allocateAndBindWithPadding(m_device, m_physDevice, {m_geoVertBuf, m_geoIdxBuf}, allocFlags);

  m_pCopyHelper->UpdateBuffer(m_geoVertBuf, 0, vertices.data(), vertexBufSize);
  m_pCopyHelper->UpdateBuffer(m_geoIdxBuf, 0, indices.data(), indexBufSize);

  if (m_useRTX)
  {
    AddBLAS(0);
  }
}


uint32_t SceneManager::AddMeshFromFile(const std::string &meshPath)
{
  //@TODO: other file formats
  auto data = cmesh::LoadMeshFromVSGF(meshPath.c_str());

  if (data.VerticesNum() == 0)
    RUN_TIME_ERROR(("can't load mesh at " + meshPath).c_str());

  return AddMeshFromData(data);
}

uint32_t SceneManager::AddMeshFromData(cmesh::SimpleMesh &meshData)
{
  assert(meshData.VerticesNum() > 0);
  assert(meshData.IndicesNum() > 0);

  m_pMeshData->Append(meshData);

  MeshInfo info;
  info.m_vertNum = meshData.VerticesNum();
  info.m_indNum = meshData.IndicesNum();

  info.m_vertexOffset = m_totalVertices;
  info.m_indexOffset = m_totalIndices;

  info.m_vertexBufOffset = info.m_vertexOffset * m_pMeshData->SingleVertexSize();
  info.m_indexBufOffset = info.m_indexOffset * m_pMeshData->SingleIndexSize();

  m_totalVertices += meshData.VerticesNum();
  m_totalIndices += meshData.IndicesNum();

  m_meshInfos.push_back(info);

  return m_meshInfos.size() - 1;
}

uint32_t SceneManager::InstanceMesh(const uint32_t meshId, const LiteMath::float4x4 &matrix, bool markForRender)
{
  assert(meshId < m_meshInfos.size());

  //@TODO: maybe move
  m_instanceMatrices.push_back(matrix);

  InstanceInfo info;
  info.inst_id = m_instanceMatrices.size() - 1;
  info.mesh_id = meshId;
  info.renderMark = markForRender;
  info.instBufOffset = (m_instanceMatrices.size() - 1) * sizeof(matrix);

  m_instanceInfos.push_back(info);

  return info.inst_id;
}

void SceneManager::MarkInstance(const uint32_t instId)
{
  assert(instId < m_instanceInfos.size());
  m_instanceInfos[instId].renderMark = true;
}

void SceneManager::UnmarkInstance(const uint32_t instId)
{
  assert(instId < m_instanceInfos.size());
  m_instanceInfos[instId].renderMark = false;
}

void SceneManager::LoadGeoDataOnGPU()
{
  VkDeviceSize vertexBufSize = m_pMeshData->VertexDataSize();
  VkDeviceSize indexBufSize = m_pMeshData->IndexDataSize();

  VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (m_useRTX)
  {
    flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }

  const VkBufferUsageFlags vertFlags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags;
  m_geoVertBuf = vk_utils::createBuffer(m_device, vertexBufSize, vertFlags);

  const VkBufferUsageFlags idxFlags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | flags;
  m_geoIdxBuf = vk_utils::createBuffer(m_device, indexBufSize, idxFlags);

  VkDeviceSize infoBufSize = m_meshInfos.size() * sizeof(uint32_t) * 2;
  m_meshInfoBuf = vk_utils::createBuffer(m_device, infoBufSize, flags);

  VkMemoryAllocateFlags allocFlags{};
  if (m_useRTX)
  {
    allocFlags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
  }

  m_geoMemAlloc = vk_utils::allocateAndBindWithPadding(m_device, m_physDevice, {m_geoVertBuf, m_geoIdxBuf, m_meshInfoBuf}, allocFlags);

  std::vector<LiteMath::uint2> mesh_info_tmp;
  for (const auto &m : m_meshInfos)
  {
    mesh_info_tmp.emplace_back(m.m_indexOffset, m.m_vertexOffset);
  }

  m_pCopyHelper->UpdateBuffer(m_geoVertBuf, 0, m_pMeshData->VertexData(), vertexBufSize);
  m_pCopyHelper->UpdateBuffer(m_geoIdxBuf, 0, m_pMeshData->IndexData(), indexBufSize);
  if (!mesh_info_tmp.empty())
    m_pCopyHelper->UpdateBuffer(m_meshInfoBuf, 0, mesh_info_tmp.data(),
                                mesh_info_tmp.size() * sizeof(mesh_info_tmp[0]));
}

void SceneManager::DrawMarkedInstances()
{

}

void SceneManager::DestroyScene()
{
  if (m_geoVertBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_geoVertBuf, nullptr);
    m_geoVertBuf = VK_NULL_HANDLE;
  }

  if (m_geoIdxBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_geoIdxBuf, nullptr);
    m_geoIdxBuf = VK_NULL_HANDLE;
  }

  if (m_instanceMatricesBuffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_instanceMatricesBuffer, nullptr);
    m_instanceMatricesBuffer = VK_NULL_HANDLE;
  }

  if (m_geoMemAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, m_geoMemAlloc, nullptr);
    m_geoMemAlloc = VK_NULL_HANDLE;
  }

  m_pCopyHelper = nullptr;

  m_meshInfos.clear();
  m_pMeshData = nullptr;
  m_instanceInfos.clear();
  m_instanceMatrices.clear();
}

void SceneManager::AddBLAS(uint32_t meshIdx)
{
  VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
  VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

  vertexBufferDeviceAddress.deviceAddress = vk_rt_utils::getBufferDeviceAddress(m_device, m_geoVertBuf);
  indexBufferDeviceAddress.deviceAddress = vk_rt_utils::getBufferDeviceAddress(m_device, m_geoIdxBuf);

  VkAccelerationStructureGeometryKHR accelerationStructureGeometry{};
  accelerationStructureGeometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
  accelerationStructureGeometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  accelerationStructureGeometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  accelerationStructureGeometry.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  accelerationStructureGeometry.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  accelerationStructureGeometry.geometry.triangles.vertexData = vertexBufferDeviceAddress;
  if (m_debug)
  {
    accelerationStructureGeometry.geometry.triangles.maxVertex = 3;
    accelerationStructureGeometry.geometry.triangles.vertexStride = sizeof(Vertex);
  }
  else
  {
    accelerationStructureGeometry.geometry.triangles.maxVertex = m_meshInfos[meshIdx].m_vertNum;
    accelerationStructureGeometry.geometry.triangles.vertexStride = m_pMeshData->SingleVertexSize();
  }

  accelerationStructureGeometry.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
  accelerationStructureGeometry.geometry.triangles.indexData = indexBufferDeviceAddress;
  accelerationStructureGeometry.geometry.triangles.transformData.deviceAddress = 0;
  accelerationStructureGeometry.geometry.triangles.transformData.hostAddress = nullptr;


  VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
  accelerationStructureBuildRangeInfo.primitiveCount = m_meshInfos[meshIdx].m_indNum / 3;
  accelerationStructureBuildRangeInfo.primitiveOffset = m_meshInfos[meshIdx].m_indexBufOffset;
  accelerationStructureBuildRangeInfo.firstVertex = m_meshInfos[meshIdx].m_vertexOffset;
  accelerationStructureBuildRangeInfo.transformOffset = 0;

  m_blasGeom.emplace_back(accelerationStructureGeometry);
  m_blasOffsetInfo.emplace_back(accelerationStructureBuildRangeInfo);
}

void SceneManager::BuildAllBLAS()
{
  auto nBlas = m_blasGeom.size();
  m_blas.resize(nBlas);

  std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(nBlas);
  for (uint32_t idx = 0; idx < nBlas; idx++)
  {
    buildInfos[idx].sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfos[idx].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfos[idx].geometryCount = 1;
    buildInfos[idx].pGeometries = &m_blasGeom[idx];
    buildInfos[idx].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfos[idx].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfos[idx].srcAccelerationStructure = VK_NULL_HANDLE;
  }

  VkDeviceSize maxScratch = 0;
  // Determine scratch buffer size depending on the largest geometry
  for (size_t idx = 0; idx < nBlas; idx++)
  {
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                                            &buildInfos[idx], &m_blasOffsetInfo[idx].primitiveCount, &sizeInfo);

    vk_rt_utils::createAccelerationStructure(m_blas[idx], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                                             sizeInfo, m_device, m_physDevice);
    maxScratch = std::max(maxScratch, sizeInfo.buildScratchSize);
  }

  vk_rt_utils::RTScratchBuffer scratchBuffer = vk_rt_utils::allocScratchBuffer(m_device, m_physDevice, maxScratch);

  VkCommandPool commandPool = vk_utils::createCommandPool(m_device, m_graphicsQId, VkCommandPoolCreateFlagBits(0));
  std::vector<VkCommandBuffer> allCmdBufs = vk_utils::createCommandBuffers(m_device, commandPool, nBlas);
  for (uint32_t idx = 0; idx < nBlas; idx++)
  {
    VkCommandBufferBeginInfo cmdBufInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    VK_CHECK_RESULT(vkBeginCommandBuffer(allCmdBufs[idx], &cmdBufInfo));
    auto &blas = m_blas[idx];
    buildInfos[idx].dstAccelerationStructure = blas.handle;
    buildInfos[idx].scratchData.deviceAddress = scratchBuffer.deviceAddress;

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR *> pBuildOffset(m_blasOffsetInfo.size());
    //    for(size_t infoIdx = 0; infoIdx < m_blasOffsetInfo.size(); infoIdx++)
    pBuildOffset[0] = &m_blasOffsetInfo[idx];

    vkCmdBuildAccelerationStructuresKHR(allCmdBufs[idx], 1, &buildInfos[idx], pBuildOffset.data());

    // barrier for scratch buffer
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    vkCmdPipelineBarrier(allCmdBufs[idx],
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    vkEndCommandBuffer(allCmdBufs[idx]);
  }
  vk_utils::executeCommandBufferNow(allCmdBufs, m_graphicsQ, m_device);
  allCmdBufs.clear();

  if (scratchBuffer.memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, scratchBuffer.memory, nullptr);
  }
  if (scratchBuffer.buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, scratchBuffer.buffer, nullptr);
  }
}

void SceneManager::BuildTLAS()
{
  std::vector<VkAccelerationStructureInstanceKHR> geometryInstances;
  geometryInstances.reserve(m_instanceInfos.size());

#ifdef USE_MANY_HIT_SHADERS
  std::map<uint32_t, uint32_t> materialMap = { {0, LAMBERT_MTL}, {1, GGX_MTL}, {2, MIRROR_MTL}, {3, BLEND_MTL}, {4, MIRROR_MTL}, {5, EMISSION_MTL} };
#endif

  for (const auto &inst : m_instanceInfos)
  {
    auto transform = transformMatrixFromFloat4x4(m_instanceMatrices[inst.inst_id]);
    VkAccelerationStructureInstanceKHR instance{};
    instance.transform = transform;
    instance.instanceCustomIndex = inst.mesh_id;
    instance.mask = 0xFF;
#ifdef USE_MANY_HIT_SHADERS
    instance.instanceShaderBindingTableRecordOffset = materialMap[inst.mesh_id];
#else
    instance.instanceShaderBindingTableRecordOffset = 0;
#endif
    instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    instance.accelerationStructureReference = m_blas[inst.mesh_id].deviceAddress;

    geometryInstances.push_back(instance);
  }

  VkBuffer instancesBuffer = VK_NULL_HANDLE;

  VkMemoryRequirements memReqs{};
  instancesBuffer = vk_utils::createBuffer(m_device,
                                           sizeof(VkAccelerationStructureInstanceKHR) * geometryInstances.size(),
                                           VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                           VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                           VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           &memReqs);

  VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
  memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

  VkDeviceMemory instancesAlloc;
  VkMemoryAllocateInfo allocateInfo = {};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.pNext = &memoryAllocateFlagsInfo;
  allocateInfo.allocationSize = memReqs.size;
  allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                                          m_physDevice);
  VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, nullptr, &instancesAlloc));

  VK_CHECK_RESULT(vkBindBufferMemory(m_device, instancesBuffer, instancesAlloc, 0));
  m_pCopyHelper->UpdateBuffer(instancesBuffer, 0, geometryInstances.data(),
                              sizeof(VkAccelerationStructureInstanceKHR) * geometryInstances.size());


  VkAccelerationStructureGeometryInstancesDataKHR instancesVk{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
  instancesVk.arrayOfPointers = VK_FALSE;
  instancesVk.data.deviceAddress = vk_rt_utils::getBufferDeviceAddress(m_device, instancesBuffer);

  VkAccelerationStructureGeometryKHR topASGeometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  topASGeometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
  topASGeometry.geometry.instances = instancesVk;

  VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
  buildInfo.geometryCount = 1;
  buildInfo.pGeometries = &topASGeometry;
  buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR; // VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
  buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
  buildInfo.srcAccelerationStructure = VK_NULL_HANDLE;

  uint32_t count = geometryInstances.size();
  VkAccelerationStructureBuildSizesInfoKHR sizeInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &count,
                                          &sizeInfo);


  vk_rt_utils::createAccelerationStructure(m_tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
                                           sizeInfo, m_device, m_physDevice);

  vk_rt_utils::RTScratchBuffer scratchBuffer = vk_rt_utils::allocScratchBuffer(m_device, m_physDevice,
                                                                               sizeInfo.buildScratchSize);

  buildInfo.srcAccelerationStructure = VK_NULL_HANDLE; //update ...
  buildInfo.dstAccelerationStructure = m_tlas.handle;
  buildInfo.scratchData.deviceAddress = scratchBuffer.deviceAddress;

  VkAccelerationStructureBuildRangeInfoKHR accelerationStructureBuildRangeInfo{};
  accelerationStructureBuildRangeInfo.primitiveCount = geometryInstances.size();
  accelerationStructureBuildRangeInfo.primitiveOffset = 0;
  accelerationStructureBuildRangeInfo.firstVertex = 0;
  accelerationStructureBuildRangeInfo.transformOffset = 0;
  std::vector<VkAccelerationStructureBuildRangeInfoKHR *> accelerationBuildStructureRangeInfos = {
      &accelerationStructureBuildRangeInfo};

  VkCommandPool commandPool = vk_utils::createCommandPool(m_device, m_graphicsQId, VkCommandPoolCreateFlagBits(0));
  VkCommandBuffer commandBuffer = vk_utils::createCommandBuffer(m_device, commandPool);
  VkCommandBufferBeginInfo cmdBufInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));
  vkCmdBuildAccelerationStructuresKHR(
      commandBuffer,
      1,
      &buildInfo,
      accelerationBuildStructureRangeInfos.data());
  vkEndCommandBuffer(commandBuffer);
  vk_utils::executeCommandBufferNow(commandBuffer, m_graphicsQ, m_device);

  if (scratchBuffer.memory != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, scratchBuffer.memory, nullptr);
  }
  if (scratchBuffer.buffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, scratchBuffer.buffer, nullptr);
  }

  if (instancesAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, instancesAlloc, nullptr);
  }
  if (instancesBuffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, instancesBuffer, nullptr);
  }
}