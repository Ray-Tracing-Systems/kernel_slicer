#pragma once
#include <vector>
#include <memory>
#include <cstdint>

#include "LiteMath.h"
#include "CrossRT.h"      // ray tracing interface
#include "BVH2CommonRT.h" // types of BVH data (BVHNode, ...): with software ray tracing they are member vectors of the generated class

using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::uint;

// Checks adding geometry (new BLAS) and instances to a scene after CommitDeviceData():
// host functions change 'm_pAccelStruct', CommitAppendedData() delivers changes to GPU.
// Render() traces orthographic rays along +Z and writes (instId, primId) of the nearest hit.
class SceneAppend
{
public:
  SceneAppend();
  virtual ~SceneAppend() { m_pAccelStruct = nullptr; }

  uint32_t AddMesh(const std::vector<float4>& a_verts, const std::vector<uint32_t>& a_indices);
  void     SetInstances(const std::vector<uint32_t>& a_geomIds, const std::vector<float4x4>& a_matrices); // ClearScene, AddInstance, CommitScene

  virtual void Render(uint32_t a_width, uint32_t a_height, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}                                       // will be overriden in generated class
  virtual void CommitAppendedData() {}                                     // will be overriden in generated class
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]) {} // will be overriden in generated class

  // initial reserves for hardware ray tracing (VulkanRTX), deliberately small: the test adds much more geometry,
  // so geometry buffers, BLAS memory and scratch buffer have to grow
  virtual void SceneRestrictions(uint32_t a_restrictions[4]) const
  {
    a_restrictions[0] = 4;    // maxMeshes
    a_restrictions[1] = 3000; // maxTotalVertices
    a_restrictions[2] = 1000; // maxTotalPrimitives
    a_restrictions[3] = 500;  // maxPrimitivesPerMesh
  }

  void kernel2D_Trace(uint32_t a_width, uint32_t a_height, uint32_t* a_out);

  std::shared_ptr<ISceneObject> m_pAccelStruct = nullptr;
  float m_sceneSize = 10.0f; // rays cover [-m_sceneSize, m_sceneSize]^2
};
