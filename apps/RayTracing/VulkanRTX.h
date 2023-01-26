#pragma once

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
  VulkanRTX(std::shared_ptr<SceneManager> a_pScnMgr);
  VulkanRTX(VkDevice a_device, VkPhysicalDevice a_physDevice, uint32_t a_graphicsQId, std::shared_ptr<vk_utils::ICopyEngine> a_pCopyHelper,
            uint32_t maxMeshes, uint32_t maxTotalVertices, uint32_t maxTotalPrimitives, uint32_t maxPrimitivesPerMesh, bool build_as_add);
  ~VulkanRTX();
  const char* Name() const override { return "VulkanRTX"; }
  
  void ClearGeom() override;
  
  uint32_t AddGeom_Triangles3f(const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber, BuildQuality a_qualityLevel, size_t vByteStride) override;
  void     UpdateGeom_Triangles3f(uint32_t a_geomId, const float* a_vpos3f, size_t a_vertNumber, const uint32_t* a_triIndices, size_t a_indNumber,  BuildQuality a_qualityLevel, size_t vByteStride) override;

  void ClearScene() override; 
  void CommitScene(BuildQuality a_qualityLevel) override; 
  
  uint32_t AddInstance(uint32_t a_geomId, const LiteMath::float4x4& a_matrix) override;
  void     UpdateInstance(uint32_t a_instanceId, const LiteMath::float4x4& a_matrix) override;

  CRT_Hit  RayQuery_NearestHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;
  bool     RayQuery_AnyHit(LiteMath::float4 posAndNear, LiteMath::float4 dirAndFar) override;

  ////////////////////////////////////////////////////////////////////////////////////////////////

  void SetSceneAccelStruct(VkAccelerationStructureKHR handle) { m_accel = handle; }
  VkAccelerationStructureKHR GetSceneAccelStruct() const { return m_accel; }
  std::shared_ptr<SceneManager> GetSceneManager() const { return m_pScnMgr; }

  static constexpr size_t VERTEX_SIZE = sizeof(float) * 4;
protected:
  VkAccelerationStructureKHR m_accel = VK_NULL_HANDLE;
  std::shared_ptr<SceneManager> m_pScnMgr;
};