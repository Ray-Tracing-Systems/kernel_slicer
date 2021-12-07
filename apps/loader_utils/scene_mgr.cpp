#include <map>
#include <array>
#include "scene_mgr.h"
#include "vk_utils.h"
#include "vk_buffers.h"

VkTransformMatrixKHR transformMatrixFromFloat4x4(const LiteMath::float4x4 &m)
{
  VkTransformMatrixKHR transformMatrix;
  for(int i = 0; i < 3; ++i)
  {
    for(int j = 0; j < 4; ++j)
    {
      transformMatrix.matrix[i][j] = m(i, j);
    }
  }
  return transformMatrix;
}

SceneManager::SceneManager(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId,
  std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper, LoaderConfig a_config) :
                m_device(a_device), m_physDevice(a_physDevice), m_graphicsQId(a_graphicsQId),
                m_pCopyHelper(a_pCopyHelper), m_config(a_config)
{
  vkGetDeviceQueue(m_device, m_graphicsQId, 0, &m_graphicsQ);

  if(m_config.build_acc_structs)
  {
//    m_pBuilder = std::make_unique<vk_rt_utils::AccelStructureBuilder>(m_device, m_physDevice, a_graphicsQId, m_graphicsQ);
    m_pBuilderV2 = std::make_unique<vk_rt_utils::AccelStructureBuilderV2>(m_device, m_physDevice, a_graphicsQId, m_graphicsQ);
  }

  m_useRTX = m_config.build_acc_structs && m_config.builder_type == BVH_BUILDER_TYPE::RTX;

  m_pool = vk_utils::createCommandPool(m_device, m_graphicsQId, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);
}


hydra_xml::Camera SceneManager::GetCamera(uint32_t camId) const
{
  if(camId >= m_sceneCameras.size())
  {
    std::stringstream ss;
    ss << "[SceneManager::GetCamera] camera with id = " << camId << " was not loaded, using default camera.";
    vk_utils::logWarning(ss.str());

    hydra_xml::Camera res = {};
    res.fov = 60;
    res.nearPlane = 0.1f;
    res.farPlane  = 1000.0f;
    res.pos[0] = 0.0f; res.pos[1] = 0.0f; res.pos[2] = 15.0f;
    res.up[0] = 0.0f; res.up[1] = 1.0f; res.up[2] = 0.0f;
    res.lookAt[0] = 0.0f; res.lookAt[1] = 0.0f; res.lookAt[2] = 0.0f;

    return res;
  }

  return m_sceneCameras[camId];
}

//void SceneManager::LoadSingleTriangle()
//{
//  std::vector<Vertex> vertices =
//  {
//    { {  1.0f,  1.0f, 0.0f } },
//    { { -1.0f,  1.0f, 0.0f } },
//    { {  0.0f, -1.0f, 0.0f } }
//  };
//
//  std::vector<uint32_t> indices = { 0, 1, 2 };
//  m_totalIndices = static_cast<uint32_t>(indices.size());
//
//  VkDeviceSize vertexBufSize = sizeof(Vertex) * vertices.size();
//  VkDeviceSize indexBufSize  = sizeof(uint32_t) * indices.size();
//
//  VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
//  if(m_useRTX)
//  {
//    flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
//             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
//  }
//
//  const VkBufferUsageFlags vertFlags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags;
//  m_geoVertBuf = vk_utils::createBuffer(m_device, vertexBufSize, vertFlags);
//
//  const VkBufferUsageFlags idxFlags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | flags;
//  m_geoIdxBuf = vk_utils::createBuffer(m_device, indexBufSize, idxFlags);
//
//  VkMemoryAllocateFlags allocFlags {};
//  if(m_useRTX)
//  {
//    allocFlags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
//  }
//
//  m_geoMemAlloc = vk_utils::allocateAndBindWithPadding(m_device, m_physDevice, {m_geoVertBuf, m_geoIdxBuf}, allocFlags);
//
//  m_pCopyHelper->UpdateBuffer(m_geoVertBuf, 0, vertices.data(),  vertexBufSize);
//  m_pCopyHelper->UpdateBuffer(m_geoIdxBuf,  0, indices.data(), indexBufSize);
//
//  if(m_config.build_acc_structs)
//  {
//    AddBLAS(0);
//  }
//}

VkPipelineVertexInputStateCreateInfo SceneManager::GetPipelineVertexInputStateCreateInfo()
{
  auto currState = m_pMeshData->VertexInputLayout();
  if(m_config.instance_matrix_as_vertex_attribute)
  {
    vk_utils::AddInstanceMatrixAttributeToVertexLayout(1, sizeof(LiteMath::float4x4), currState);
  }
  return currState;
}

uint32_t SceneManager::AddMeshFromFile(const std::string& meshPath)
{
  //@TODO: other file formats
  auto data = cmesh::LoadMeshFromVSGF(meshPath.c_str());

  if(data.VerticesNum() == 0)
    RUN_TIME_ERROR(("can't load mesh at " + meshPath).c_str());

  return AddMeshFromData(data);
}

uint32_t SceneManager::AddMeshFromData(cmesh::SimpleMesh &meshData)
{
  assert(meshData.VerticesNum() > 0);
  assert(meshData.IndicesNum() > 0);

  m_pMeshData->Append(meshData);
  auto old_size = m_matIDs.size();
  m_matIDs.resize(m_matIDs.size() + meshData.matIndices.size());
  std::copy(meshData.matIndices.begin(), meshData.matIndices.end(), m_matIDs.begin() + old_size);

  MeshInfo info;
  info.m_vertNum = meshData.VerticesNum();
  info.m_indNum  = meshData.IndicesNum();

  info.m_vertexOffset = m_totalVertices;
  info.m_indexOffset  = m_totalIndices;

  info.m_vertexBufOffset = info.m_vertexOffset * m_pMeshData->SingleVertexSize();
  info.m_indexBufOffset  = info.m_indexOffset  * m_pMeshData->SingleIndexSize();

  m_totalVertices += meshData.VerticesNum();
  m_totalIndices  += meshData.IndicesNum();

  m_meshInfos.push_back(info);

  return m_meshInfos.size() - 1;
}

uint32_t SceneManager::AddMeshFromDataAndQueueBuildAS(cmesh::SimpleMesh &meshData)
{
  auto idx = AddMeshFromData(meshData);
  LoadOneMeshOnGPU(idx);
  if(m_config.build_acc_structs)
  {
    AddBLAS(idx);
  }

  return idx;
}

uint32_t SceneManager::InstanceMesh(const uint32_t meshId, const LiteMath::float4x4 &matrix, bool markForRender)
{
  assert(meshId < m_meshInfos.size());

  //@TODO: maybe move
  m_instanceMatrices.push_back(matrix);

  InstanceInfo info;
  info.inst_id       = m_instanceMatrices.size() - 1;
  info.mesh_id       = meshId;
  info.renderMark    = markForRender;
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

void SceneManager::InitMeshCPU(MESH_FORMATS format)
{
  if(format == MESH_FORMATS::MESH_8F)
    m_pMeshData = std::make_shared<Mesh8F>();
  else if(format == MESH_FORMATS::MESH_4F)
    m_pMeshData = std::make_shared<Mesh4F>();
}

void SceneManager::InitGeoBuffersGPU(uint32_t a_meshNum, uint32_t a_totalVertNum, uint32_t a_totalIndicesNum)
{
  VkDeviceSize vertexBufSize = m_pMeshData->SingleVertexSize() * a_totalVertNum;
  VkDeviceSize indexBufSize  = m_pMeshData->SingleIndexSize() * a_totalIndicesNum;

  VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if(m_useRTX)
  {
    flags |= VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  }

  std::vector<VkBuffer> all_buffers;

  const VkBufferUsageFlags vertFlags = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | flags;
  m_geoVertBuf = vk_utils::createBuffer(m_device, vertexBufSize, vertFlags);
  all_buffers.push_back(m_geoVertBuf);

  const VkBufferUsageFlags idxFlags = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | flags;
  m_geoIdxBuf = vk_utils::createBuffer(m_device, indexBufSize, idxFlags);
  all_buffers.push_back(m_geoIdxBuf);

  VkDeviceSize infoBufSize = a_meshNum * sizeof(uint32_t) * 2;
  m_meshInfoBuf = vk_utils::createBuffer(m_device, infoBufSize, flags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  all_buffers.push_back(m_meshInfoBuf);

  VkDeviceSize matIdsBufSize = (a_totalIndicesNum / 3) * sizeof(uint32_t);
  m_matIdsBuf = vk_utils::createBuffer(m_device, matIdsBufSize, flags | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  all_buffers.push_back(m_matIdsBuf);

  VkMemoryAllocateFlags allocFlags {};
  if(m_useRTX)
  {
    allocFlags |= VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;
  }

  m_geoMemAlloc = vk_utils::allocateAndBindWithPadding(m_device, m_physDevice, all_buffers, allocFlags);
}

void SceneManager::LoadOneMeshOnGPU(uint32_t meshIdx)
{
  VkDeviceSize vertexBufSize = m_meshInfos[meshIdx].m_vertNum * m_pMeshData->SingleVertexSize();
  VkDeviceSize indexBufSize  = m_meshInfos[meshIdx].m_indNum  * m_pMeshData->SingleIndexSize();

  auto vertSrc = m_pMeshData->VertexData() + m_loadedVertices * (m_pMeshData->SingleVertexSize() / sizeof(float));
  auto indSrc  = m_pMeshData->IndexData() + m_loadedIndices;
  auto loadedPrims = (m_loadedIndices / 3);
  m_pCopyHelper->UpdateBuffer(m_geoVertBuf, m_loadedVertices * m_pMeshData->SingleVertexSize(), vertSrc, vertexBufSize);
  m_pCopyHelper->UpdateBuffer(m_geoIdxBuf, m_loadedIndices * m_pMeshData->SingleIndexSize(), indSrc, indexBufSize);
  m_pCopyHelper->UpdateBuffer(m_matIdsBuf,  loadedPrims * sizeof(uint32_t),
    m_matIDs.data() + loadedPrims, (m_meshInfos[meshIdx].m_indNum / 3) * sizeof(m_matIDs[0]));

//  if(meshIdx == 8)
//  {
//    std::ofstream file("tmp.txt");
//    for(size_t i = 0; i < m_meshInfos[meshIdx].m_indNum / 3; ++i)
//    {
//      file << m_matIDs[loadedPrims + i] << "\n";
//    }
//    file.close();
//  }
  m_loadedVertices += m_meshInfos[meshIdx].m_vertNum ;
  m_loadedIndices  += m_meshInfos[meshIdx].m_indNum;
}

void SceneManager::LoadCommonGeoDataOnGPU()
{
//  VkDeviceSize vertexBufSize = m_pMeshData->VertexDataSize();
//  VkDeviceSize indexBufSize  = m_pMeshData->IndexDataSize();
//  VkDeviceSize instMatBufSize = m_instanceMatrices.size() * sizeof(m_instanceMatrices[0]);

  std::vector<LiteMath::uint2> mesh_info_tmp;
  for(const auto& m : m_meshInfos)
  {
    mesh_info_tmp.emplace_back(m.m_indexOffset, m.m_vertexOffset);
  }

//  m_pCopyHelper->UpdateBuffer(m_geoVertBuf, 0, m_pMeshData->VertexData(), vertexBufSize);
//  m_pCopyHelper->UpdateBuffer(m_geoIdxBuf,  0, m_pMeshData->IndexData(), indexBufSize);
//  if(m_config.instance_matrix_as_vertex_attribute)
//  {
//    m_pCopyHelper->UpdateBuffer(m_instMatricesBuf, 0, m_instanceMatrices.data(), instMatBufSize);
//  }
  if(!mesh_info_tmp.empty())
  {
    m_pCopyHelper->UpdateBuffer(m_meshInfoBuf, 0, mesh_info_tmp.data(), mesh_info_tmp.size() * sizeof(mesh_info_tmp[0]));
  }
}

void SceneManager::LoadInstanceDataOnGPU()
{
  VkDeviceSize instMatBufSize = m_instanceMatrices.size() * sizeof(m_instanceMatrices[0]);
  VkBufferUsageFlags flags = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;

  m_instMatricesBuf = vk_utils::createBuffer(m_device, instMatBufSize, flags);
  m_instMemAlloc    = vk_utils::allocateAndBindWithPadding(m_device, m_physDevice, {m_instMatricesBuf});

  m_pCopyHelper->UpdateBuffer(m_instMatricesBuf, 0, m_instanceMatrices.data(), instMatBufSize);
}


void SceneManager::DrawMarkedInstances()
{

}

SceneManager::~SceneManager()
{
  DestroyScene();
  if(m_pool != VK_NULL_HANDLE)
  {
    vkDestroyCommandPool(m_device, m_pool, nullptr);
    m_pool = VK_NULL_HANDLE;
  }

}

void SceneManager::DestroyScene()
{
  if(m_geoVertBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_geoVertBuf, nullptr);
    m_geoVertBuf = VK_NULL_HANDLE;
  }

  if(m_geoIdxBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_geoIdxBuf, nullptr);
    m_geoIdxBuf = VK_NULL_HANDLE;
  }

  if(m_meshInfoBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_meshInfoBuf, nullptr);
    m_meshInfoBuf = VK_NULL_HANDLE;
  }

  if(m_matIdsBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_matIdsBuf, nullptr);
    m_matIdsBuf = VK_NULL_HANDLE;
  }

  if(m_geoMemAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, m_geoMemAlloc, nullptr);
    m_geoMemAlloc = VK_NULL_HANDLE;
  }

  if(m_instMatricesBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_instMatricesBuf, nullptr);
    m_instMatricesBuf = VK_NULL_HANDLE;
  }

  if(m_instMemAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, m_instMemAlloc, nullptr);
    m_instMemAlloc = VK_NULL_HANDLE;
  }

  if(m_materialBuf != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, m_materialBuf, nullptr);
    m_materialBuf = VK_NULL_HANDLE;
  }
  if(m_matMemAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, m_matMemAlloc, nullptr);
    m_matMemAlloc = VK_NULL_HANDLE;
  }

  for(auto& [_, tex] : m_texturesById)
  {
    if(tex.view != VK_NULL_HANDLE)
      vkDestroyImageView(m_device, tex.view, nullptr);

    if(tex.image != VK_NULL_HANDLE)
      vkDestroyImage(m_device, tex.image, nullptr);
  }
  if(m_texturesMemAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, m_texturesMemAlloc, nullptr);
    m_texturesMemAlloc = VK_NULL_HANDLE;
  }

  {
    std::sort(m_samplers.begin(), m_samplers.end());
    auto last = std::unique(m_samplers.begin(), m_samplers.end());
    m_samplers.erase(last, m_samplers.end());
    for(auto& samp : m_samplers)
    {
      if(samp != VK_NULL_HANDLE)
      {
        vkDestroySampler(m_device, samp, nullptr);
      }
    }
  }

  if(m_config.build_acc_structs)
    m_pBuilderV2->Destroy();

  m_loadedVertices        = 0;
  m_loadedIndices         = 0;
  m_totalVertices = 0u;
  m_totalIndices  = 0u;
  m_meshInfos.clear();
  m_pMeshData = nullptr;
  m_instanceInfos.clear();
  m_instanceMatrices.clear();
  m_matIDs.clear();

  m_textureViews.clear();
  m_samplers.clear();
  m_texturesById.clear();
  m_sceneCameras.clear();
}

void SceneManager::AddBLAS(uint32_t meshIdx)
{
  VkDeviceOrHostAddressConstKHR vertexBufferDeviceAddress{};
  VkDeviceOrHostAddressConstKHR indexBufferDeviceAddress{};

  vertexBufferDeviceAddress.deviceAddress = vk_rt_utils::getBufferDeviceAddress(m_device, m_geoVertBuf);
  indexBufferDeviceAddress.deviceAddress  = vk_rt_utils::getBufferDeviceAddress(m_device, m_geoIdxBuf);

  m_pBuilderV2->AddBLAS(m_meshInfos[meshIdx], m_pMeshData->SingleVertexSize(),
    vertexBufferDeviceAddress, indexBufferDeviceAddress);
}

void SceneManager::BuildAllBLAS()
{
//  m_pBuilder->BuildBLAS(m_blasData);
  m_pBuilderV2->BuildAllBLAS();
}

void SceneManager::BuildTLAS()
{
  BuildAllBLAS();

  std::vector<VkAccelerationStructureInstanceKHR> geometryInstances;
  geometryInstances.reserve(m_instanceInfos.size());

#ifdef USE_MANY_HIT_SHADERS
  std::map<uint32_t, uint32_t> materialMap = { {0, LAMBERT_MTL}, {1, GGX_MTL}, {2, MIRROR_MTL}, {3, BLEND_MTL}, {4, MIRROR_MTL}, {5, EMISSION_MTL} };
#endif

  for(const auto& inst : m_instanceInfos)
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
    instance.accelerationStructureReference = m_pBuilderV2->GetBLASDeviceAddress(inst.mesh_id);//m_blas[inst.mesh_id].deviceAddress;

    geometryInstances.push_back(instance);
  }

  VkBuffer instancesBuffer = VK_NULL_HANDLE;

  VkMemoryRequirements memReqs {};
  instancesBuffer = vk_utils::createBuffer(m_device, sizeof(VkAccelerationStructureInstanceKHR) * geometryInstances.size(),
    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    &memReqs);

  VkMemoryAllocateFlagsInfo memoryAllocateFlagsInfo{};
  memoryAllocateFlagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
  memoryAllocateFlagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT_KHR;

  VkDeviceMemory instancesAlloc;
  VkMemoryAllocateInfo allocateInfo = {};
  allocateInfo.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.pNext           = &memoryAllocateFlagsInfo;
  allocateInfo.allocationSize  = memReqs.size;
  allocateInfo.memoryTypeIndex = vk_utils::findMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, m_physDevice);
  VK_CHECK_RESULT(vkAllocateMemory(m_device, &allocateInfo, nullptr, &instancesAlloc));

  VK_CHECK_RESULT(vkBindBufferMemory(m_device, instancesBuffer, instancesAlloc, 0));
  m_pCopyHelper->UpdateBuffer(instancesBuffer, 0, geometryInstances.data(),
    sizeof(VkAccelerationStructureInstanceKHR) * geometryInstances.size());

  VkDeviceOrHostAddressConstKHR instBufferDeviceAddress{};
  instBufferDeviceAddress.deviceAddress = vk_rt_utils::getBufferDeviceAddress(m_device, instancesBuffer);
  m_pBuilderV2->BuildTLAS(geometryInstances.size(), instBufferDeviceAddress);

  if (instancesAlloc != VK_NULL_HANDLE)
  {
    vkFreeMemory(m_device, instancesAlloc, nullptr);
  }
  if (instancesBuffer != VK_NULL_HANDLE)
  {
    vkDestroyBuffer(m_device, instancesBuffer, nullptr);
  }
}

bool SceneManager::InitEmptyScene(uint32_t maxMeshes, uint32_t maxTotalVertices, uint32_t maxTotalPrimitives, uint32_t maxPrimitivesPerMesh)
{
  InitMeshCPU(m_config.mesh_format);

  InitGeoBuffersGPU(maxMeshes, maxTotalVertices, maxTotalPrimitives * 3);
  if(m_config.build_acc_structs)
  {
    m_pBuilderV2->Init(maxTotalVertices, maxPrimitivesPerMesh, maxTotalPrimitives, m_pMeshData->SingleVertexSize());
  }

  return true;
}