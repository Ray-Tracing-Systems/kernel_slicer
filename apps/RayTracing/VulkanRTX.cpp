#include <iostream>
#include <vector>
#include <unordered_map>
#include <cassert>
#include <limits>

#include "CrossRT.h"
#include "scene_mgr.h" // RTX implementation of acceleration structures

class VulkanRTX : public ISceneObject
{
public:
  VulkanRTX();
  ~VulkanRTX();
  void ClearGeom() override;
  
  uint32_t AddGeom_Triangles4f(const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber) override;
  void     UpdateGeom_Triangles4f(uint32_t a_geomId, const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber) override;

  void ClearScene() override; 
  void CommitScene  () override; 
  
  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override;
  void     UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override;

  CRT_Hit  RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;

protected:
  std::shared_ptr<SceneManager> m_pScnMgr;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

VulkanRTX::VulkanRTX()
{
  m_pScnMgr = nullptr; //#TODO: get vulkan objects here in some way
  //m_pScnMgr = std::make_shared<SceneManager>(device, physicalDevice, queueComputeFID, queueComputeFID, true); 
}

VulkanRTX::~VulkanRTX()
{
  m_pScnMgr = nullptr;
}

void VulkanRTX::ClearGeom()
{
  
}
  
uint32_t VulkanRTX::AddGeom_Triangles4f(const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{ 
  cmesh::SimpleMesh meshData(a_vertNumber, a_indNumber);
  memcpy(meshData.vPos4f.data(), a_vpos4f, a_vertNumber*sizeof(LiteMath::float4));
  memcpy(meshData.indices.data(), a_triIndices, a_indNumber*sizeof(uint32_t));
  return m_pScnMgr->AddMeshFromData(meshData);
}

void VulkanRTX::UpdateGeom_Triangles4f(uint32_t a_geomId, const LiteMath::float4* a_vpos4f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber)
{
  std::cout << "[VulkanRTX::UpdateGeom_Triangles4f]: not implemented" << std::endl;
}

void VulkanRTX::ClearScene()
{
 
} 

uint32_t VulkanRTX::AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix)
{
  return m_pScnMgr->InstanceMesh(a_geomId, a_matrix);
}

void VulkanRTX::CommitScene()
{
  m_pScnMgr->BuildAllBLAS(); // why can't we just build BVH tree for single mesh in thsi API, this seems impractical
  m_pScnMgr->BuildTLAS();
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

ISceneObject* CreateVulkanRTX() { return new VulkanRTX(); }