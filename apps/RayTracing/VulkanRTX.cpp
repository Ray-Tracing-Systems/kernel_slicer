#include "VulkanRTX.h"
#include "ray_tracing/vk_rt_utils.h"
#include "vk_utils.h"

ISceneObject* CreateVulkanRTX(std::shared_ptr<SceneManager> a_pScnMgr) { return new VulkanRTX(a_pScnMgr); }

ISceneObject* CreateVulkanRTX(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId, std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper,
                              uint32_t a_maxMeshes, uint32_t a_maxTotalVertices, uint32_t a_maxTotalPrimitives, uint32_t a_maxPrimitivesPerMesh,
                              bool build_as_add)
{
  static constexpr uint64_t STAGING_MEM_SIZE = 16 * 16 * 1024u;
  VkQueue queue;
  vkGetDeviceQueue(a_device, a_graphicsQId, 0, &queue);

  auto copyHelper = std::make_shared<vk_utils::PingPongCopyHelper>(a_physDevice, a_device, queue, a_graphicsQId, STAGING_MEM_SIZE);

  return new VulkanRTX(a_device, a_physDevice, a_graphicsQId, copyHelper,
    a_maxMeshes, a_maxTotalVertices, a_maxTotalPrimitives, a_maxPrimitivesPerMesh, build_as_add);
}

VulkanRTX::VulkanRTX(std::shared_ptr<SceneManager> a_pScnMgr) : m_pScnMgr(a_pScnMgr)
{
}

VulkanRTX::VulkanRTX(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId, std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper,
                     uint32_t a_maxMeshes, uint32_t a_maxTotalVertices, uint32_t a_maxTotalPrimitives, uint32_t a_maxPrimitivesPerMesh,
                     bool build_as_add)
{
  LoaderConfig conf = {};
  conf.load_geometry = true;
  conf.load_materials = MATERIAL_LOAD_MODE::NONE;
  conf.build_acc_structs = true;
  conf.build_acc_structs_while_loading_scene = build_as_add;
  conf.builder_type = BVH_BUILDER_TYPE::RTX;
  conf.mesh_format  = MESH_FORMATS::MESH_4F;

  m_pScnMgr = std::make_shared<SceneManager>(a_device, a_physDevice, a_graphicsQId, a_pCopyHelper, conf);
  m_pScnMgr->InitEmptyScene(a_maxMeshes, a_maxTotalVertices, a_maxTotalPrimitives, a_maxPrimitivesPerMesh);
}


VulkanRTX::~VulkanRTX()
{
  m_pScnMgr = nullptr;
}

void VulkanRTX::ClearGeom()
{
//  m_pScnMgr->DestroyScene();
}
  
uint32_t VulkanRTX::AddGeom_Triangles4f(const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{
  cmesh::SimpleMesh mesh(a_vertNumber, a_indNumber);

  memcpy(mesh.vPos4f.data(), (float*)a_vpos4f, a_vertNumber * VERTEX_SIZE);
  memcpy(mesh.indices.data(), a_triIndices, a_indNumber * sizeof(a_triIndices[0]));

  auto idx = m_pScnMgr->AddMeshFromDataAndQueueBuildAS(mesh);

  return idx;
}

void VulkanRTX::UpdateGeom_Triangles4f(uint32_t a_geomId, const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{
  std::cout << "[VulkanRTX::UpdateGeom_Triangles4f]: not implemented" << std::endl;
}

void VulkanRTX::ClearScene()
{
  std::cout << "[VulkanRTX::ClearScene]: not implemented" << std::endl;
} 

uint32_t VulkanRTX::AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix)
{
  return m_pScnMgr->InstanceMesh(a_geomId, a_matrix);
}

void VulkanRTX::CommitScene()
{
  m_pScnMgr->BuildTLAS();
  m_accel = m_pScnMgr->GetTLAS();
}  

void VulkanRTX::UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix)
{
  std::cout << "[VulkanRTX::UpdateInstance]: not implemented" << std::endl;
}

CRT_Hit VulkanRTX::RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar)
{    
  CRT_Hit result;
  result.t      = std::numeric_limits<float>::max();
  result.geomId = uint32_t(-1);
  result.instId = uint32_t(-1);
  result.primId = uint32_t(-1);
  return result;
}

bool VulkanRTX::RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar)
{
  return false;
}