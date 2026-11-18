#pragma once
#include <vector>
#include <cstdint>

#include "LiteMath.h"
#include "Image2d.h"
#include "raster.h"

using LiteMath::float2;
using LiteMath::float3;
using LiteMath::float4;
using LiteMath::float4x4;
using LiteMath::int2;
using LiteMath::clamp;
using LiteMath::normalize;
using LiteMath::dot;
using LiteMath::length;
using LiteImage::Image2D;

struct ShadeVaryings  // vertex shader output, pixel shader input
{
  float4 normal;      ///< world space
  float4 worldPos;
  float4 color;
};

/**
\brief A field of many animated objects: three meshes (cube, sphere, torus), each is drawn by one
       DrawIndexedInstancedTriangles; the ground plane is an ordinary DrawIndexedTriangles in the same pass.

Vertices of all meshes are stored in one set of vertex buffers, every mesh has its own index buffer.
Instances of all meshes are stored in one set of instance buffers: instances of mesh 'k' occupy
[m_firstInstance[k], m_firstInstance[k] + m_instanceNum[k]), the vertex shader gets the first instance as an argument.
*/
class InstancedField
{
public:
  InstancedField() {}
  virtual ~InstancedField() {}

  void SetViewport(int a_width, int a_height); ///< host only: geometry, instances, camera and pipeline state

  virtual void Render(int a_width, int a_height, float a_time, uint32_t* a_out [[size("a_width*a_height")]]);

  virtual void CommitDeviceData() {}
  virtual void GetExecutionTime(const char* a_funcName, float a_out[4]);

  static constexpr uint32_t MESH_CUBE   = 0;
  static constexpr uint32_t MESH_SPHERE = 1;
  static constexpr uint32_t MESH_TORUS  = 2;
  static constexpr uint32_t MESH_NUM    = 3;

protected:

  float4 kernelVS_Object(uint32_t a_vertId, uint32_t a_instId, uint32_t a_firstInstance, float a_time, ShadeVaryings* a_out) const;
  float4 kernelVS_Ground(uint32_t a_vertId, ShadeVaryings* a_out) const;
  float4 kernelPS_Object(const ShadeVaryings& a_in) const;
  float4 kernelPS_Ground(const ShadeVaryings& a_in) const;
  virtual void kernel2D_ToOutput(int a_width, int a_height, uint32_t* a_out);

  // vertices of all meshes
  //
  std::vector<float4>   m_positions;
  std::vector<float4>   m_normals;

  // index buffers of meshes, indices refer to the common vertex buffers
  //
  std::vector<uint32_t> m_groundIndices;
  std::vector<uint32_t> m_cubeIndices;
  std::vector<uint32_t> m_sphereIndices;
  std::vector<uint32_t> m_torusIndices;

  // instances of all meshes
  //
  std::vector<float4>   m_instPosScale;  ///< (x, y, z) -- position on the ground, w -- scale
  std::vector<float4>   m_instAnim;      ///< x -- rotation speed, y -- phase, z -- bounce height
  std::vector<float4>   m_instColor;
  uint32_t              m_firstInstance[MESH_NUM];
  uint32_t              m_instanceNum[MESH_NUM];

  float4x4              m_viewProj;
  float4                m_camPos;
  float4                m_lightDir;
  float4                m_fogColor;

  Image2D<uint32_t>     m_color;     ///< RGBA8
  Image2D<float>        m_depth;
  LiteRS::Pipeline      m_pipeline;
  float                 m_timeRender = 0.0f;
};
